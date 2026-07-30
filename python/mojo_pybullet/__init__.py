"""A focused PyBullet-compatible primitive collision and dynamics engine."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import math
import subprocess
import threading
import time
import warnings
from typing import Sequence

import numpy as np

from ._lib import addr, lib

DIRECT = 2
GUI = 1
SHARED_MEMORY = 3

GEOM_SPHERE = 2
GEOM_BOX = 3
GEOM_PLANE = 6

LINK_FRAME = 1
WORLD_FRAME = 2

_ZERO3 = (0.0, 0.0, 0.0)
_IDENTITY4 = (0.0, 0.0, 0.0, 1.0)


@dataclass
class _Shape:
    kind: int
    data: tuple[float, float, float, float]


def _vec(values: Sequence[float], n: int, name: str) -> np.ndarray:
    value = np.asarray(values, dtype=np.float64)
    if value.shape != (n,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain {n} finite values")
    return value


def _finite_float(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _quat(values: Sequence[float]) -> np.ndarray:
    value = _vec(values, 4, "orientation")
    norm = float(np.linalg.norm(value))
    if norm == 0.0:
        raise ValueError("orientation cannot be the zero quaternion")
    return value / norm


def _rotation(q: Sequence[float]) -> np.ndarray:
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


class _World:
    def __init__(self) -> None:
        # ctypes releases the GIL while Mojo runs.  Array replacement must not
        # invalidate a cached address until the call using it has returned.
        self._ffi_lock = threading.RLock()
        self.shapes: list[_Shape] = []
        self.gravity = np.zeros(3, dtype=np.float64)
        self.dt = 1.0 / 240.0
        self.iterations = 4
        library = lib()
        self._closest_kernel = library.mpb_closest_points
        self._ray_kernel = library.mpb_ray_test_batch
        self._ray_gpu_kernel = library.mpb_ray_test_batch_gpu
        self._step_kernel = library.mpb_step
        self.reset_arrays()

    def reset_arrays(self) -> None:
        with self._ffi_lock:
            self._active_flags: list[bool] = []
            self.types = np.empty(0, dtype=np.int64)
            self.data = np.empty((0, 4), dtype=np.float64)
            self.pos = np.empty((0, 3), dtype=np.float64)
            self.orn = np.empty((0, 4), dtype=np.float64)
            self.vel = np.empty((0, 3), dtype=np.float64)
            self.angvel = np.empty((0, 3), dtype=np.float64)
            self.force = np.empty((0, 3), dtype=np.float64)
            self.torque = np.empty((0, 3), dtype=np.float64)
            self.inv_mass = np.empty(0, dtype=np.float64)
            self.mass = np.empty(0, dtype=np.float64)
            self.inertia = np.empty((0, 3), dtype=np.float64)
            self.inv_inertia = np.empty((0, 3), dtype=np.float64)
            self.restitution = np.empty(0, dtype=np.float64)
            self.friction = np.empty(0, dtype=np.float64)
            self.linear_damping = np.empty(0, dtype=np.float64)
            self.angular_damping = np.empty(0, dtype=np.float64)
            self.active = np.empty(0, dtype=np.int64)
            self.ids = np.empty(0, dtype=np.int64)
            self._step_scratch = (ctypes.c_double * 10)()
            self._step_scratch_addr = ctypes.addressof(self._step_scratch)
            self._refresh_ffi_addresses()

    def _refresh_ffi_addresses(self) -> None:
        buffers = (
            self.types,
            self.data,
            self.pos,
            self.orn,
            self.vel,
            self.angvel,
            self.force,
            self.torque,
            self.inv_mass,
            self.inv_inertia,
            self.restitution,
            self.friction,
            self.linear_damping,
            self.angular_damping,
            self.active,
        )
        if any(not array.flags.c_contiguous for array in buffers):
            raise RuntimeError("internal FFI buffers must be C-contiguous")
        if self.types.dtype != np.int64 or self.active.dtype != np.int64:
            raise RuntimeError("internal integer FFI buffers must use int64")
        if any(array.dtype != np.float64 for array in buffers[1:-1]):
            raise RuntimeError("internal floating-point FFI buffers must use float64")
        self._step_addresses = tuple(addr(array) for array in buffers)
        self._collision_addresses = self._step_addresses[:4]
        self._ray_world_addresses = (
            *self._collision_addresses,
            addr(self.ids),
            addr(self.active),
        )

    @property
    def count(self) -> int:
        return len(self._active_flags)

    def body_index(self, body: int) -> int:
        if not isinstance(body, (int, np.integer)) or body < 0 or body >= self.count:
            raise ValueError(f"invalid bodyUniqueId {body}")
        if not self._active_flags[body]:
            raise ValueError(f"bodyUniqueId {body} has been removed")
        return int(body)

    def add_body(
        self,
        shape: _Shape,
        mass: float,
        position: np.ndarray,
        orientation: np.ndarray,
    ) -> int:
        with self._ffi_lock:
            body = self.count
            if shape.kind == GEOM_SPHERE:
                inertia = np.full(3, 0.4 * mass * shape.data[0] ** 2)
            elif shape.kind == GEOM_BOX:
                hx, hy, hz, _ = shape.data
                inertia = mass / 3.0 * np.array(
                    [hy * hy + hz * hz, hx * hx + hz * hz, hx * hx + hy * hy]
                )
            else:
                inertia = np.zeros(3)
            inverse_inertia = np.divide(
                1.0, inertia, out=np.zeros(3), where=inertia > 0.0
            )
            self.types = np.append(self.types, np.int64(shape.kind))
            self.data = np.vstack((self.data, shape.data))
            self.pos = np.vstack((self.pos, position))
            self.orn = np.vstack((self.orn, orientation))
            self.vel = np.vstack((self.vel, np.zeros(3)))
            self.angvel = np.vstack((self.angvel, np.zeros(3)))
            self.force = np.vstack((self.force, np.zeros(3)))
            self.torque = np.vstack((self.torque, np.zeros(3)))
            self.mass = np.append(self.mass, mass)
            self.inv_mass = np.append(self.inv_mass, 1.0 / mass if mass > 0 else 0.0)
            self.inertia = np.vstack((self.inertia, inertia))
            self.inv_inertia = np.vstack((self.inv_inertia, inverse_inertia))
            self.restitution = np.append(self.restitution, 0.0)
            self.friction = np.append(self.friction, 0.5)
            self.linear_damping = np.append(self.linear_damping, 0.04)
            self.angular_damping = np.append(self.angular_damping, 0.04)
            self.active = np.append(self.active, np.int64(1))
            self.ids = np.arange(body + 1, dtype=np.int64)
            self._active_flags.append(True)
            self._refresh_ffi_addresses()
            return body


_clients: dict[int, _World] = {}
_next_client = 0


def _world(physicsClientId: int = 0) -> _World:
    try:
        return _clients[int(physicsClientId)]
    except KeyError as exc:
        raise RuntimeError(f"Not connected to physics server {physicsClientId}") from exc


def connect(connectionMode: int, options: str = "") -> int:
    del options
    if connectionMode != DIRECT:
        raise NotImplementedError("mojo-pybullet supports DIRECT connections only")
    global _next_client
    client = _next_client
    _next_client += 1
    _clients[client] = _World()
    return client


def disconnect(physicsClientId: int = 0) -> None:
    _clients.pop(int(physicsClientId), None)


def isConnected(physicsClientId: int = 0) -> int:
    return int(int(physicsClientId) in _clients)


def getConnectionInfo(physicsClientId: int = 0) -> dict[str, int]:
    connected = isConnected(physicsClientId)
    return {"isConnected": connected, "connectionMethod": DIRECT if connected else 0}


def getAPIVersion() -> int:
    return 202010061


def resetSimulation(physicsClientId: int = 0) -> None:
    world = _world(physicsClientId)
    with world._ffi_lock:
        world.shapes.clear()
        world.reset_arrays()
        world.gravity[:] = 0.0
        world.dt = 1.0 / 240.0
        world.iterations = 4


def createCollisionShape(
    shapeType: int,
    radius: float = 0.5,
    halfExtents: Sequence[float] = (1.0, 1.0, 1.0),
    height: float = 1.0,
    fileName: str = "",
    meshScale: Sequence[float] = (1.0, 1.0, 1.0),
    planeNormal: Sequence[float] = (0.0, 0.0, 1.0),
    flags: int = 0,
    collisionFramePosition: Sequence[float] = (0.0, 0.0, 0.0),
    collisionFrameOrientation: Sequence[float] = (0.0, 0.0, 0.0, 1.0),
    physicsClientId: int = 0,
) -> int:
    del height, fileName, meshScale, flags
    if not np.allclose(collisionFramePosition, (0.0, 0.0, 0.0)) or not np.allclose(
        collisionFrameOrientation, (0.0, 0.0, 0.0, 1.0)
    ):
        raise NotImplementedError("non-identity collision frames are not covered")
    if shapeType == GEOM_SPHERE:
        radius = _finite_float(radius, "radius")
        if radius <= 0:
            raise ValueError("radius must be positive")
        data = (radius, 0.0, 0.0, 0.0)
    elif shapeType == GEOM_BOX:
        half = _vec(halfExtents, 3, "halfExtents")
        if np.any(half <= 0):
            raise ValueError("halfExtents must be positive")
        data = (float(half[0]), float(half[1]), float(half[2]), 0.0)
    elif shapeType == GEOM_PLANE:
        normal = _vec(planeNormal, 3, "planeNormal")
        length = float(np.linalg.norm(normal))
        if length == 0:
            raise ValueError("planeNormal cannot be zero")
        normal /= length
        data = (float(normal[0]), float(normal[1]), float(normal[2]), 0.0)
    else:
        raise NotImplementedError("covered shape types are GEOM_SPHERE, GEOM_BOX, GEOM_PLANE")
    world = _world(physicsClientId)
    shape = len(world.shapes)
    world.shapes.append(_Shape(int(shapeType), data))
    return shape


def createMultiBody(
    baseMass: float = 0.0,
    baseCollisionShapeIndex: int = -1,
    baseVisualShapeIndex: int = -1,
    basePosition: Sequence[float] = (0.0, 0.0, 0.0),
    baseOrientation: Sequence[float] = (0.0, 0.0, 0.0, 1.0),
    baseInertialFramePosition: Sequence[float] = (0.0, 0.0, 0.0),
    baseInertialFrameOrientation: Sequence[float] = (0.0, 0.0, 0.0, 1.0),
    linkMasses: Sequence[float] | None = None,
    linkCollisionShapeIndices: Sequence[int] | None = None,
    linkVisualShapeIndices: Sequence[int] | None = None,
    linkPositions: Sequence[Sequence[float]] | None = None,
    linkOrientations: Sequence[Sequence[float]] | None = None,
    linkInertialFramePositions: Sequence[Sequence[float]] | None = None,
    linkInertialFrameOrientations: Sequence[Sequence[float]] | None = None,
    linkParentIndices: Sequence[int] | None = None,
    linkJointTypes: Sequence[int] | None = None,
    linkJointAxis: Sequence[Sequence[float]] | None = None,
    useMaximalCoordinates: int = 0,
    flags: int = -1,
    batchPositions: Sequence[Sequence[float]] | None = None,
    physicsClientId: int = 0,
) -> int:
    del baseVisualShapeIndex, useMaximalCoordinates, flags
    link_arguments = (
        linkMasses,
        linkCollisionShapeIndices,
        linkVisualShapeIndices,
        linkPositions,
        linkOrientations,
        linkInertialFramePositions,
        linkInertialFrameOrientations,
        linkParentIndices,
        linkJointTypes,
        linkJointAxis,
    )
    if any(value not in (None, []) for value in link_arguments):
        raise NotImplementedError("articulated links are outside the covered subset")
    if batchPositions is not None:
        raise NotImplementedError("batchPositions is not covered")
    baseMass = _finite_float(baseMass, "baseMass")
    if baseMass < 0:
        raise ValueError("baseMass cannot be negative")
    if not np.allclose(baseInertialFramePosition, (0.0, 0.0, 0.0)) or not np.allclose(
        baseInertialFrameOrientation, (0.0, 0.0, 0.0, 1.0)
    ):
        raise NotImplementedError("offset inertial frames are not covered")
    world = _world(physicsClientId)
    if baseCollisionShapeIndex < 0 or baseCollisionShapeIndex >= len(world.shapes):
        raise ValueError(f"invalid baseCollisionShapeIndex {baseCollisionShapeIndex}")
    return world.add_body(
        world.shapes[baseCollisionShapeIndex],
        baseMass,
        _vec(basePosition, 3, "basePosition"),
        _quat(baseOrientation),
    )


def removeBody(bodyUniqueId: int, physicsClientId: int = 0) -> None:
    world = _world(physicsClientId)
    with world._ffi_lock:
        body = world.body_index(bodyUniqueId)
        world.active[body] = 0
        world._active_flags[body] = False


def getNumBodies(physicsClientId: int = 0) -> int:
    return int(_world(physicsClientId).active.sum())


def getBodyUniqueId(serialIndex: int, physicsClientId: int = 0) -> int:
    ids = np.flatnonzero(_world(physicsClientId).active)
    if serialIndex < 0 or serialIndex >= len(ids):
        raise ValueError("serialIndex out of range")
    return int(ids[serialIndex])


def getBasePositionAndOrientation(
    bodyUniqueId: int, physicsClientId: int = 0
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    world = _world(physicsClientId)
    body = world.body_index(bodyUniqueId)
    return tuple(world.pos[body].tolist()), tuple(world.orn[body].tolist())


def resetBasePositionAndOrientation(
    bodyUniqueId: int,
    posObj: Sequence[float],
    ornObj: Sequence[float],
    physicsClientId: int = 0,
) -> None:
    world = _world(physicsClientId)
    body = world.body_index(bodyUniqueId)
    world.pos[body] = _vec(posObj, 3, "posObj")
    world.orn[body] = _quat(ornObj)


def getBaseVelocity(
    bodyUniqueId: int, physicsClientId: int = 0
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    world = _world(physicsClientId)
    body = world.body_index(bodyUniqueId)
    return tuple(world.vel[body].tolist()), tuple(world.angvel[body].tolist())


def resetBaseVelocity(
    objectUniqueId: int,
    linearVelocity: Sequence[float] = (0.0, 0.0, 0.0),
    angularVelocity: Sequence[float] = (0.0, 0.0, 0.0),
    physicsClientId: int = 0,
) -> None:
    world = _world(physicsClientId)
    body = world.body_index(objectUniqueId)
    world.vel[body] = _vec(linearVelocity, 3, "linearVelocity")
    world.angvel[body] = _vec(angularVelocity, 3, "angularVelocity")


def setGravity(x: float, y: float, z: float, physicsClientId: int = 0) -> None:
    _world(physicsClientId).gravity[:] = _vec((x, y, z), 3, "gravity")


def setTimeStep(timeStep: float, physicsClientId: int = 0) -> None:
    timeStep = _finite_float(timeStep, "timeStep")
    if timeStep <= 0:
        raise ValueError("timeStep must be positive")
    _world(physicsClientId).dt = timeStep


def setPhysicsEngineParameter(physicsClientId: int = 0, **kwargs: float) -> None:
    world = _world(physicsClientId)
    supported = {"fixedTimeStep", "numSolverIterations"}
    unknown = set(kwargs) - supported
    if unknown:
        raise NotImplementedError(f"unsupported physics parameters: {sorted(unknown)}")
    if "fixedTimeStep" in kwargs:
        setTimeStep(float(kwargs["fixedTimeStep"]), physicsClientId)
    if "numSolverIterations" in kwargs:
        value = kwargs["numSolverIterations"]
        if not isinstance(value, (int, np.integer)):
            raise TypeError("numSolverIterations must be an integer")
        world.iterations = max(1, int(value))


def stepSimulation(physicsClientId: int = 0) -> None:
    world = _world(physicsClientId)
    with world._ffi_lock:
        if world.count == 0:
            return
        world._step_kernel(
            *world._step_addresses,
            world.count,
            *world.gravity,
            world.dt,
            world.iterations,
            world._step_scratch_addr,
        )


def changeDynamics(bodyUniqueId: int, linkIndex: int, physicsClientId: int = 0, **kwargs) -> None:
    if linkIndex != -1:
        raise NotImplementedError("only base linkIndex=-1 is covered")
    world = _world(physicsClientId)
    body = world.body_index(bodyUniqueId)
    names = {
        "lateralFriction": world.friction,
        "restitution": world.restitution,
        "linearDamping": world.linear_damping,
        "angularDamping": world.angular_damping,
    }
    unknown = set(kwargs) - (set(names) | {"mass"})
    if unknown:
        raise NotImplementedError(f"unsupported dynamics properties: {sorted(unknown)}")
    for name, array in names.items():
        if name in kwargs:
            value = _finite_float(kwargs[name], name)
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
            array[body] = value
    if "mass" in kwargs:
        mass = _finite_float(kwargs["mass"], "mass")
        if mass < 0:
            raise ValueError("mass cannot be negative")
        old_mass = world.mass[body]
        world.mass[body] = mass
        world.inv_mass[body] = 1.0 / mass if mass else 0.0
        if old_mass > 0:
            world.inertia[body] *= mass / old_mass
        else:
            shape = _Shape(int(world.types[body]), tuple(world.data[body]))
            if shape.kind == GEOM_SPHERE:
                world.inertia[body] = 0.4 * mass * shape.data[0] ** 2
            elif shape.kind == GEOM_BOX:
                hx, hy, hz, _ = shape.data
                world.inertia[body] = mass / 3 * np.array(
                    [
                        hy * hy + hz * hz,
                        hx * hx + hz * hz,
                        hx * hx + hy * hy,
                    ]
                )
        world.inv_inertia[body] = np.divide(
            1.0, world.inertia[body], out=np.zeros(3), where=world.inertia[body] > 0
        )


def getDynamicsInfo(bodyUniqueId: int, linkIndex: int, physicsClientId: int = 0):
    if linkIndex != -1:
        raise NotImplementedError("only base linkIndex=-1 is covered")
    world = _world(physicsClientId)
    body = world.body_index(bodyUniqueId)
    return (
        float(world.mass[body]),
        float(world.friction[body]),
        tuple(world.inertia[body].tolist()),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
        float(world.restitution[body]),
        0.0,
        0.0,
        -1.0,
        -1.0,
        2,
        0.001,
    )


def applyExternalForce(
    objectUniqueId: int,
    linkIndex: int,
    forceObj: Sequence[float],
    posObj: Sequence[float],
    flags: int,
    physicsClientId: int = 0,
) -> None:
    if linkIndex != -1:
        raise NotImplementedError("only base linkIndex=-1 is covered")
    world = _world(physicsClientId)
    body = world.body_index(objectUniqueId)
    force = _vec(forceObj, 3, "forceObj")
    point = _vec(posObj, 3, "posObj")
    if flags == LINK_FRAME:
        rotation = _rotation(world.orn[body])
        force = rotation @ force
        point = world.pos[body] + rotation @ point
    elif flags != WORLD_FRAME:
        raise ValueError("flags must be WORLD_FRAME or LINK_FRAME")
    world.force[body] += force
    world.torque[body] += np.cross(point - world.pos[body], force)


def applyExternalTorque(
    objectUniqueId: int,
    linkIndex: int,
    torqueObj: Sequence[float],
    flags: int,
    physicsClientId: int = 0,
) -> None:
    if linkIndex != -1:
        raise NotImplementedError("only base linkIndex=-1 is covered")
    world = _world(physicsClientId)
    body = world.body_index(objectUniqueId)
    torque = _vec(torqueObj, 3, "torqueObj")
    if flags == LINK_FRAME:
        torque = _rotation(world.orn[body]) @ torque
    elif flags != WORLD_FRAME:
        raise ValueError("flags must be WORLD_FRAME or LINK_FRAME")
    world.torque[body] += torque


class _ContactScratch(threading.local):
    def __init__(self) -> None:
        values = (ctypes.c_double * 10)()
        self.storage = (values, ctypes.addressof(values))


_contact_scratch = _ContactScratch()
_gpu_memory_status = (0.0, False)


def _contact(world: _World, bodyA: int, bodyB: int):
    result, result_address = _contact_scratch.storage
    with world._ffi_lock:
        supported = world._closest_kernel(
            *world._collision_addresses,
            bodyA,
            bodyB,
            result_address,
        )
    if not supported:
        raise NotImplementedError("closest points cover sphere-sphere, sphere-box, sphere-plane")
    return result


def _contact_tuple(bodyA: int, bodyB: int, result: Sequence[float]):
    return (
        0,
        bodyA,
        bodyB,
        -1,
        -1,
        (float(result[0]), float(result[1]), float(result[2])),
        (float(result[3]), float(result[4]), float(result[5])),
        (float(result[6]), float(result[7]), float(result[8])),
        float(result[9]),
        0.0,
        0.0,
        (0.0, 0.0, 0.0),
        0.0,
        (0.0, 0.0, 0.0),
    )


def getClosestPoints(
    bodyA: int,
    bodyB: int,
    distance: float,
    linkIndexA: int = -1,
    linkIndexB: int = -1,
    collisionShapePositionA: Sequence[float] = _ZERO3,
    collisionShapeOrientationA: Sequence[float] = _IDENTITY4,
    collisionShapePositionB: Sequence[float] = _ZERO3,
    collisionShapeOrientationB: Sequence[float] = _IDENTITY4,
    physicsClientId: int = 0,
):
    if linkIndexA != -1 or linkIndexB != -1:
        raise NotImplementedError("only base links are covered")
    identity = (
        collisionShapePositionA is _ZERO3
        and collisionShapePositionB is _ZERO3
        and collisionShapeOrientationA is _IDENTITY4
        and collisionShapeOrientationB is _IDENTITY4
    )
    if not identity:
        identity = (
            tuple(collisionShapePositionA) == _ZERO3
            and tuple(collisionShapePositionB) == _ZERO3
            and tuple(collisionShapeOrientationA) == _IDENTITY4
            and tuple(collisionShapeOrientationB) == _IDENTITY4
        )
    if not identity:
        raise NotImplementedError("collision-shape transform overrides are not covered")
    world = _world(physicsClientId)
    if (
        type(bodyA) is int
        and 0 <= bodyA < world.count
        and world._active_flags[bodyA]
    ):
        a = bodyA
    else:
        a = world.body_index(bodyA)
    if (
        type(bodyB) is int
        and 0 <= bodyB < world.count
        and world._active_flags[bodyB]
    ):
        b = bodyB
    else:
        b = world.body_index(bodyB)
    result, result_address = _contact_scratch.storage
    with world._ffi_lock:
        supported = world._closest_kernel(
            *world._collision_addresses,
            a,
            b,
            result_address,
        )
    if not supported:
        raise NotImplementedError("closest points cover sphere-sphere, sphere-box, sphere-plane")
    if result[9] > distance:
        return ()
    return (
        (
            0,
            a,
            b,
            -1,
            -1,
            (result[0], result[1], result[2]),
            (result[3], result[4], result[5]),
            (result[6], result[7], result[8]),
            result[9],
            0.0,
            0.0,
            _ZERO3,
            0.0,
            _ZERO3,
        ),
    )


def getContactPoints(
    bodyA: int = -1,
    bodyB: int = -1,
    linkIndexA: int = -2,
    linkIndexB: int = -2,
    physicsClientId: int = 0,
):
    if linkIndexA not in (-2, -1) or linkIndexB not in (-2, -1):
        raise NotImplementedError("only base links are covered")
    world = _world(physicsClientId)
    if bodyA >= 0:
        world.body_index(bodyA)
    if bodyB >= 0:
        world.body_index(bodyB)
    contacts = []
    for a in range(world.count):
        if not world.active[a] or (bodyA >= 0 and a != bodyA):
            continue
        for b in range(world.count):
            if b == a or not world.active[b] or (bodyB >= 0 and b != bodyB):
                continue
            if bodyA < 0 and bodyB < 0 and b < a:
                continue
            try:
                result = _contact(world, a, b)
            except NotImplementedError:
                continue
            if result[9] <= 0.0:
                contacts.append(_contact_tuple(a, b, result))
    return tuple(contacts)


def _gpu_memory_ready() -> bool:
    global _gpu_memory_status
    now = time.monotonic()
    checked_at, ready = _gpu_memory_status
    if now - checked_at < 1.0:
        return ready
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
        ready = any(int(line.strip()) >= 4000 for line in result.stdout.splitlines())
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        ready = False
    _gpu_memory_status = (time.monotonic(), ready)
    return ready


def rayTestBatch(
    rayFromPositions: Sequence[Sequence[float]],
    rayToPositions: Sequence[Sequence[float]],
    numThreads: int = 1,
    parentObjectUniqueId: int = -1,
    parentLinkIndex: int = -1,
    reportHitNumber: int = -1,
    collisionFilterMask: int = -1,
    physicsClientId: int = 0,
    device: str = "cpu",
):
    del numThreads, collisionFilterMask
    if device not in ("cpu", "gpu"):
        raise ValueError("device must be 'cpu' or 'gpu'")
    if parentObjectUniqueId != -1 or parentLinkIndex != -1 or reportHitNumber != -1:
        raise NotImplementedError("parent transforms and multi-hit rays are not covered")
    starts = np.ascontiguousarray(rayFromPositions, dtype=np.float64)
    ends = np.ascontiguousarray(rayToPositions, dtype=np.float64)
    if starts.ndim != 2 or starts.shape[1:] != (3,) or starts.shape != ends.shape:
        raise ValueError("rayFromPositions and rayToPositions must both have shape (n, 3)")
    if not np.all(np.isfinite(starts)) or not np.all(np.isfinite(ends)):
        raise ValueError("ray positions must contain only finite values")
    world = _world(physicsClientId)
    count = len(starts)
    hit_ids = np.empty(count, dtype=np.int64)
    fractions = np.empty(count, dtype=np.float64)
    points = np.empty((count, 3), dtype=np.float64)
    normals = np.empty((count, 3), dtype=np.float64)
    if count:
        with world._ffi_lock:
            arguments = (
                addr(starts),
                addr(ends),
                count,
                *world._ray_world_addresses,
                world.count,
                addr(hit_ids),
                addr(fractions),
                addr(points),
                addr(normals),
            )
            gpu_attempted = (
                device == "gpu"
                and 112 * (count + world.count) < 2_000_000_000
                and _gpu_memory_ready()
            )
            used_gpu = gpu_attempted and bool(world._ray_gpu_kernel(*arguments))
            if gpu_attempted and not used_gpu:
                warnings.warn(
                    "Mojo GPU ray execution failed; falling back to CPU",
                    RuntimeWarning,
                    stacklevel=2,
                )
            if not used_gpu:
                world._ray_kernel(*arguments)
    return tuple(
        (body, -1, fraction, tuple(point), tuple(normal))
        for body, fraction, point, normal in zip(
            hit_ids.tolist(), fractions.tolist(), points.tolist(), normals.tolist()
        )
    )


def rayTest(
    rayFromPosition: Sequence[float],
    rayToPosition: Sequence[float],
    collisionFilterMask: int = -1,
    physicsClientId: int = 0,
    device: str = "cpu",
):
    return rayTestBatch(
        [rayFromPosition],
        [rayToPosition],
        collisionFilterMask=collisionFilterMask,
        physicsClientId=physicsClientId,
        device=device,
    )


def getAABB(bodyUniqueId: int, linkIndex: int = -1, physicsClientId: int = 0):
    if linkIndex != -1:
        raise NotImplementedError("only base linkIndex=-1 is covered")
    world = _world(physicsClientId)
    body = world.body_index(bodyUniqueId)
    kind = world.types[body]
    if kind == GEOM_PLANE:
        return ((-1e30, -1e30, -1e30), (1e30, 1e30, 1e30))
    if kind == GEOM_SPHERE:
        extent = np.full(3, world.data[body, 0])
    else:
        extent = np.abs(_rotation(world.orn[body])) @ world.data[body, :3]
    return (
        tuple((world.pos[body] - extent).tolist()),
        tuple((world.pos[body] + extent).tolist()),
    )


def getOverlappingObjects(aabbMin, aabbMax, physicsClientId: int = 0):
    low = _vec(aabbMin, 3, "aabbMin")
    high = _vec(aabbMax, 3, "aabbMax")
    hits = []
    world = _world(physicsClientId)
    for body in range(world.count):
        if not world.active[body]:
            continue
        body_low, body_high = getAABB(body, physicsClientId=physicsClientId)
        if np.all(np.asarray(body_high) >= low) and np.all(np.asarray(body_low) <= high):
            hits.append((body, -1))
    return tuple(hits) if hits else None


def getQuaternionFromEuler(euler: Sequence[float]) -> tuple[float, float, float, float]:
    roll, pitch, yaw = _vec(euler, 3, "euler")
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def getEulerFromQuaternion(quaternion: Sequence[float]) -> tuple[float, float, float]:
    x, y, z, w = _quat(quaternion)
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def getMatrixFromQuaternion(quaternion: Sequence[float]) -> tuple[float, ...]:
    return tuple(_rotation(_quat(quaternion)).ravel())


def multiplyTransforms(positionA, orientationA, positionB, orientationB):
    pa, qa = _vec(positionA, 3, "positionA"), _quat(orientationA)
    pb, qb = _vec(positionB, 3, "positionB"), _quat(orientationB)
    ax, ay, az, aw = qa
    bx, by, bz, bw = qb
    q = (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )
    return tuple((pa + _rotation(qa) @ pb).tolist()), q


def invertTransform(position, orientation):
    p = _vec(position, 3, "position")
    x, y, z, w = _quat(orientation)
    inverse = (-x, -y, -z, w)
    return tuple((-_rotation(inverse) @ p).tolist()), inverse


__all__ = [
    name
    for name in globals()
    if not name.startswith("_")
    and name not in {"ctypes", "math", "np", "Sequence", "subprocess", "threading", "time", "warnings"}
]
