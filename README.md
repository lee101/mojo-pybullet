# mojo-pybullet

`mojo-pybullet` is a focused port of PyBullet's primitive collision detection and
base rigid-body dynamics to Mojo. It provides a Python module named
`mojo_pybullet` whose covered functions use PyBullet's names, argument order,
keyword names, constants, and result tuple layouts.

This is useful for lightweight headless primitive worlds and large batches of
rays without embedding the full Bullet engine. It is not a replacement for all
of PyBullet.

## Covered subset

- `DIRECT` clients: `connect`, `disconnect`, `isConnected`,
  `getConnectionInfo`, and `resetSimulation`
- sphere, oriented box, and plane collision shapes
- base-only bodies: creation, removal, poses, velocities, AABBs, and overlapping
  AABB queries
- `getClosestPoints` for sphere-sphere, sphere-box, and sphere-plane pairs,
  including PyBullet's contact witness and normal conventions
- `getContactPoints` for the same primitive pairs
- `rayTest` and `rayTestBatch` against spheres, oriented boxes, and two-sided
  planes
- gravity, time step, base forces and torques, configurable damping/material
  values, and discrete time stepping
- quaternion/Euler matrices and transform helpers

Dynamics resolve sphere-sphere, sphere-box, and sphere-plane contacts. The
solver uses semi-implicit Euler integration and sequential impulses. It includes
speed-dependent damping, friction impulses, restitution, and penetration
correction, but is not Bullet's full rigid-body solver.

Not covered are GUI/shared-memory servers, visual shapes, meshes, heightfields,
capsules, cylinders, articulated links, joints, constraints, soft bodies,
plugins, rendering, file loading, collision groups, continuous collision
detection, and box-box dynamics. Unsupported covered-adjacent requests raise
`NotImplementedError`; they are not approximated silently.

## Install and run

The Pixi environment contains the pinned Mojo nightly, NumPy, pytest, and the
real conda-forge PyBullet build used by the parity suite.

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` creates `dist/libmojo-pybullet.so`.

## Usage

```python
import mojo_pybullet as p

client = p.connect(p.DIRECT)
p.setGravity(0, 0, -9.8, physicsClientId=client)

floor_shape = p.createCollisionShape(
    p.GEOM_PLANE, planeNormal=(0, 0, 1), physicsClientId=client
)
ball_shape = p.createCollisionShape(
    p.GEOM_SPHERE, radius=0.5, physicsClientId=client
)
floor = p.createMultiBody(
    0, floor_shape, basePosition=(0, 0, 0), physicsClientId=client
)
ball = p.createMultiBody(
    1, ball_shape, basePosition=(0, 0, 2), physicsClientId=client
)

for _ in range(240):
    p.stepSimulation(physicsClientId=client)

position, orientation = p.getBasePositionAndOrientation(ball, physicsClientId=client)
print(position)  # approximately (0.0, 0.0, 0.5)
print(p.rayTest((0, 0, 3), (0, 0, -1), physicsClientId=client))
p.disconnect(client)
```

CPU execution is the default. Large ray batches can explicitly request the
optional GPU path with `p.rayTestBatch(starts, ends, device="gpu")`. It falls
back to CPU when the GPU is unavailable or the request is too large. A GPU
execution failure emits `RuntimeWarning` before falling back.

Run the checked-in version with
`pixi run python examples/quickstart.py`.

## Correctness

The test suite runs the Mojo implementation and conda-forge PyBullet on
identical worlds. Its 31 tests assert numerical parity for contact positions,
normals and signed distances; rotated AABBs; ray hit IDs, fractions, points and
normals; transforms; free-fall trajectories; external forces; dynamics
properties; and resting sphere-plane contact. They also cover a SIMD remainder,
both sides of the parallel integration threshold, the GPU ray path, FFI input
validation, and pose/velocity/reset lifecycle behavior. Box comparisons account
for Bullet's convex margin.

```bash
pixi run test
# 31 passed
```

## Benchmarks

Measured with `pixi run bench` on this machine immediately before publication.
The benchmark reported `x86_64, Linux 6.8.0-136-generic`. Times are the best of
five warm runs. Each row exercises the public Python API and therefore includes
result tuple construction and ctypes overhead.

| Case | mojo-pybullet | PyBullet | Relative |
|---|---:|---:|---:|
| `rayTestBatch`, 8,192 rays x 64 bodies | 14.22 ms | 23.32 ms | 1.64x faster |
| `rayTestBatch` GPU, 8,192 rays x 64 bodies | 9.18 ms | 24.33 ms | 2.65x faster |
| `getClosestPoints`, 20,000 calls | 109.23 ms | 130.27 ms | 1.19x faster |
| `stepSimulation`, 128 bodies x 100 steps | 8.78 ms | 12.13 ms | 1.38x faster |

The CPU ray kernel handles the complete batch in one compiled call and assigns
batches of at least 2,048 rays to coarse 256-ray parallel tasks. The optional
GPU ray kernel benefits from reusing body data across many independent rays.
Closest-point calls reuse
thread-local result storage and cached zero-copy buffer addresses. Dynamics
uses a sphere overlap fast reject, exits solver iterations when no contacts
remain, and parallelizes sufficiently large independent body integration.
Force and torque clearing uses native-width float64 SIMD with a scalar tail.

## How it works

Python owns contiguous NumPy structure-of-arrays buffers for shape types, shape
parameters, transforms, velocities, accumulated forces, inverse mass and
inertia, and material properties. A world body ID is the stable row index;
removed rows are masked rather than renumbered.

The Python layer passes buffer addresses, scalar sizes, and scalar
settings through `ctypes`. The single Mojo compilation unit reconstructs
`UnsafePointer[..., AnyOrigin[mut=True]]` values inside its C ABI exports.
Before crossing the boundary, Python validates shapes, finite inputs, C
contiguity, and exact `float64`/`int64` dtypes. A per-world lock keeps every
NumPy buffer alive and prevents replacement while Mojo is using its address.
The CPU paths write results directly into NumPy-owned buffers.
`stepSimulation` updates world buffers in place, clears accumulated
forces, detects supported primitive contacts, and applies sequential impulses.
The explicit GPU ray path uses short-lived device buffers and copies results
back into the same NumPy-owned output arrays.

## Development

```bash
pixi run build
pixi run test
pixi run bench
```

Benchmarks should always be run through the Pixi task; it holds a machine-wide
file lock to avoid overlapping benchmark jobs.
