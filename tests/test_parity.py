import math

import numpy as np
import pybullet as bullet
import pytest

import mojo_pybullet as mojo


@pytest.fixture
def clients():
    ours = mojo.connect(mojo.DIRECT)
    theirs = bullet.connect(bullet.DIRECT)
    try:
        yield ours, theirs
    finally:
        mojo.disconnect(ours)
        bullet.disconnect(theirs)


def make_shape_pair(clients, kind, **kwargs):
    ours, theirs = clients
    return (
        mojo.createCollisionShape(kind, physicsClientId=ours, **kwargs),
        bullet.createCollisionShape(kind, physicsClientId=theirs, **kwargs),
    )


def make_body_pair(clients, shapes, mass=0.0, position=(0, 0, 0), orientation=(0, 0, 0, 1)):
    ours, theirs = clients
    return (
        mojo.createMultiBody(mass, shapes[0], -1, position, orientation, physicsClientId=ours),
        bullet.createMultiBody(mass, shapes[1], -1, position, orientation, physicsClientId=theirs),
    )


def assert_vec(got, expected, atol=1e-10):
    assert np.asarray(got) == pytest.approx(np.asarray(expected), abs=atol)


def assert_closest_equal(got, expected, atol=1e-9):
    assert len(got) == len(expected)
    if not got:
        return
    a, b = got[0], expected[0]
    assert a[:5] == b[:5]
    assert_vec(a[5], b[5], atol)
    assert_vec(a[6], b[6], atol)
    assert_vec(a[7], b[7], atol)
    assert a[8] == pytest.approx(b[8], abs=atol)


def test_connection_and_body_lifecycle():
    client = mojo.connect(mojo.DIRECT)
    assert mojo.isConnected(client) == 1
    assert mojo.getConnectionInfo(client) == {
        "isConnected": 1,
        "connectionMethod": mojo.DIRECT,
    }
    shape = mojo.createCollisionShape(mojo.GEOM_SPHERE, radius=0.5, physicsClientId=client)
    bodies = [
        mojo.createMultiBody(0, shape, physicsClientId=client),
        mojo.createMultiBody(1, shape, physicsClientId=client),
    ]
    assert mojo.getNumBodies(client) == 2
    assert [mojo.getBodyUniqueId(i, client) for i in range(2)] == bodies
    mojo.removeBody(bodies[0], client)
    assert mojo.getNumBodies(client) == 1
    assert mojo.getBodyUniqueId(0, client) == bodies[1]
    mojo.disconnect(client)
    assert mojo.isConnected(client) == 0


def test_pose_velocity_and_reset_simulation():
    client = mojo.connect(mojo.DIRECT)
    try:
        shape = mojo.createCollisionShape(
            mojo.GEOM_SPHERE, radius=0.5, physicsClientId=client
        )
        body = mojo.createMultiBody(1, shape, physicsClientId=client)
        position = (1.0, -2.0, 3.0)
        orientation = mojo.getQuaternionFromEuler((0.1, 0.2, 0.3))
        mojo.resetBasePositionAndOrientation(body, position, orientation, client)
        mojo.resetBaseVelocity(body, (4, 5, 6), (-1, -2, -3), client)
        assert_vec(mojo.getBasePositionAndOrientation(body, client)[0], position)
        assert_vec(mojo.getBasePositionAndOrientation(body, client)[1], orientation)
        assert mojo.getBaseVelocity(body, client) == (
            (4.0, 5.0, 6.0),
            (-1.0, -2.0, -3.0),
        )
        mojo.resetSimulation(client)
        assert mojo.getNumBodies(client) == 0
    finally:
        mojo.disconnect(client)


def test_invalid_numeric_inputs_are_rejected_before_ffi():
    client = mojo.connect(mojo.DIRECT)
    try:
        with pytest.raises(ValueError):
            mojo.createCollisionShape(
                mojo.GEOM_SPHERE, radius=math.nan, physicsClientId=client
            )
        with pytest.raises(ValueError):
            mojo.setGravity(0, 0, math.inf, physicsClientId=client)
        with pytest.raises(ValueError):
            mojo.rayTestBatch(((0, 0, 0),), ((math.nan, 0, 0),), physicsClientId=client)
        with pytest.raises(TypeError):
            mojo.setPhysicsEngineParameter(
                physicsClientId=client, numSolverIterations=1.5
            )
    finally:
        mojo.disconnect(client)


@pytest.mark.parametrize("separation", [3.5, 2.0, 1.25, 0.5])
def test_sphere_sphere_closest_points_match_pybullet(clients, separation):
    shapes = make_shape_pair(clients, mojo.GEOM_SPHERE, radius=1.0)
    a = make_body_pair(clients, shapes, position=(0, 0, 0))
    b = make_body_pair(clients, shapes, position=(separation, 0, 0))
    ours, theirs = clients
    assert_closest_equal(
        mojo.getClosestPoints(a[0], b[0], 5.0, physicsClientId=ours),
        bullet.getClosestPoints(a[1], b[1], 5.0, physicsClientId=theirs),
    )


@pytest.mark.parametrize(
    "sphere_position,euler",
    [
        ((2.2, 0.3, -0.4), (0.0, 0.0, 0.0)),
        ((2.0, 1.1, 0.4), (0.2, -0.4, 0.7)),
        ((0.2, -2.4, 1.0), (-0.3, 0.1, -0.5)),
    ],
)
def test_sphere_oriented_box_closest_points_match_pybullet(clients, sphere_position, euler):
    sphere = make_shape_pair(clients, mojo.GEOM_SPHERE, radius=0.35)
    box = make_shape_pair(clients, mojo.GEOM_BOX, halfExtents=(1.0, 1.5, 0.75))
    orientation = mojo.getQuaternionFromEuler(euler)
    orientation_ref = bullet.getQuaternionFromEuler(euler)
    sphere_body = make_body_pair(clients, sphere, position=sphere_position)
    box_body = make_body_pair(
        clients, box, position=(0.1, -0.2, 0.3), orientation=orientation
    )
    assert_vec(orientation, orientation_ref)
    ours, theirs = clients
    assert_closest_equal(
        mojo.getClosestPoints(
            sphere_body[0], box_body[0], 10.0, physicsClientId=ours
        ),
        bullet.getClosestPoints(
            sphere_body[1], box_body[1], 10.0, physicsClientId=theirs
        ),
        atol=2e-8,
    )
    assert_closest_equal(
        mojo.getClosestPoints(
            box_body[0], sphere_body[0], 10.0, physicsClientId=ours
        ),
        bullet.getClosestPoints(
            box_body[1], sphere_body[1], 10.0, physicsClientId=theirs
        ),
        atol=2e-8,
    )


def test_sphere_plane_closest_points_match_pybullet(clients):
    plane = make_shape_pair(clients, mojo.GEOM_PLANE, planeNormal=(0, 0, 1))
    sphere = make_shape_pair(clients, mojo.GEOM_SPHERE, radius=0.4)
    orientation = mojo.getQuaternionFromEuler((0.2, -0.3, 0.1))
    pbody = make_body_pair(
        clients, plane, position=(0.0, 0.0, -0.2), orientation=orientation
    )
    sbody = make_body_pair(clients, sphere, position=(0.2, -0.1, 1.3))
    ours, theirs = clients
    assert_closest_equal(
        mojo.getClosestPoints(sbody[0], pbody[0], 5, physicsClientId=ours),
        bullet.getClosestPoints(sbody[1], pbody[1], 5, physicsClientId=theirs),
        atol=2e-9,
    )


def test_distance_threshold_and_contacts(clients):
    shapes = make_shape_pair(clients, mojo.GEOM_SPHERE, radius=1)
    a = make_body_pair(clients, shapes, position=(0, 0, 0))
    b = make_body_pair(clients, shapes, position=(1.5, 0, 0))
    ours, _ = clients
    assert mojo.getClosestPoints(a[0], b[0], -0.6, physicsClientId=ours) == ()
    assert len(mojo.getClosestPoints(a[0], b[0], 0.0, physicsClientId=ours)) == 1
    contacts = mojo.getContactPoints(a[0], b[0], physicsClientId=ours)
    assert len(contacts) == 1
    assert contacts[0][8] == pytest.approx(-0.5)


@pytest.mark.parametrize("other_kind", [mojo.GEOM_BOX, mojo.GEOM_PLANE])
def test_contact_points_cover_other_supported_pairs(clients, other_kind):
    sphere = make_shape_pair(clients, mojo.GEOM_SPHERE, radius=0.5)
    kwargs = (
        {"halfExtents": (1, 1, 1)}
        if other_kind == mojo.GEOM_BOX
        else {"planeNormal": (0, 0, 1)}
    )
    other = make_shape_pair(clients, other_kind, **kwargs)
    sphere_body = make_body_pair(clients, sphere, position=(0, 0, 0.25))
    other_body = make_body_pair(clients, other)
    ours, _ = clients
    contacts = mojo.getContactPoints(
        sphere_body[0], other_body[0], physicsClientId=ours
    )
    assert len(contacts) == 1
    assert contacts[0][8] < 0.0


def test_aabb_matches_pybullet_for_rotated_box_and_sphere(clients):
    sphere = make_shape_pair(clients, mojo.GEOM_SPHERE, radius=0.7)
    box = make_shape_pair(clients, mojo.GEOM_BOX, halfExtents=(1, 2, 0.5))
    q = mojo.getQuaternionFromEuler((0.3, -0.5, 0.8))
    bodies = [
        make_body_pair(clients, sphere, position=(1, -2, 0.5)),
        make_body_pair(clients, box, position=(-1, 0.2, 2), orientation=q),
    ]
    ours, theirs = clients
    for a, b in bodies:
        got = mojo.getAABB(a, physicsClientId=ours)
        expected = bullet.getAABB(b, physicsClientId=theirs)
        assert_vec(got[0], expected[0], 2e-9)
        assert_vec(got[1], expected[1], 2e-9)


def test_ray_test_batch_matches_pybullet(clients):
    sphere = make_shape_pair(clients, mojo.GEOM_SPHERE, radius=0.7)
    box = make_shape_pair(clients, mojo.GEOM_BOX, halfExtents=(0.5, 1.0, 1.5))
    make_body_pair(clients, sphere, position=(-1.5, 0, 0.2))
    make_body_pair(
        clients,
        box,
        position=(1.2, 0.1, 0.0),
        orientation=mojo.getQuaternionFromEuler((0.2, 0.4, -0.3)),
    )
    starts = np.array(
        [[-4, 0, 0.2], [4, 0.1, 0], [0, -4, 0], [0, 4, 3], [0, 0, 4]],
        dtype=float,
    )
    ends = -starts
    ours, theirs = clients
    got = mojo.rayTestBatch(starts, ends, physicsClientId=ours)
    expected = bullet.rayTestBatch(starts, ends, physicsClientId=theirs)
    assert len(got) == len(expected)
    for a, b in zip(got, expected):
        assert a[:2] == b[:2]
        assert a[2] == pytest.approx(b[2], abs=2e-6)
        assert_vec(a[3], b[3], 2e-6)
        assert_vec(a[4], b[4], 2e-6)


def test_gpu_ray_path_matches_cpu_or_falls_back():
    client = mojo.connect(mojo.DIRECT)
    try:
        sphere = mojo.createCollisionShape(
            mojo.GEOM_SPHERE, radius=0.7, physicsClientId=client
        )
        mojo.createMultiBody(
            0, sphere, basePosition=(0, 0, 0), physicsClientId=client
        )
        box = mojo.createCollisionShape(
            mojo.GEOM_BOX, halfExtents=(0.25, 0.5, 0.4), physicsClientId=client
        )
        mojo.createMultiBody(
            0,
            box,
            basePosition=(1.2, 0, 0),
            baseOrientation=mojo.getQuaternionFromEuler((0.1, 0.2, 0.3)),
            physicsClientId=client,
        )
        plane = mojo.createCollisionShape(
            mojo.GEOM_PLANE, planeNormal=(0, 0, 1), physicsClientId=client
        )
        mojo.createMultiBody(
            0, plane, basePosition=(0, 0, -1), physicsClientId=client
        )
        starts = np.column_stack(
            (np.linspace(-2, 2, 257), np.zeros(257), np.full(257, 2.0))
        )
        ends = starts.copy()
        ends[:, 2] = -2.0
        cpu = mojo.rayTestBatch(starts, ends, physicsClientId=client)
        gpu = mojo.rayTestBatch(starts, ends, physicsClientId=client, device="gpu")
        assert len(cpu) == len(gpu)
        for expected, got in zip(cpu, gpu):
            assert got[:2] == expected[:2]
            assert got[2] == pytest.approx(expected[2], abs=1e-12)
            assert_vec(got[3], expected[3], 1e-12)
            assert_vec(got[4], expected[4], 1e-12)
    finally:
        mojo.disconnect(client)


def test_plane_rays_from_both_sides_match_pybullet(clients):
    plane = make_shape_pair(clients, mojo.GEOM_PLANE, planeNormal=(0, 0, 1))
    make_body_pair(clients, plane)
    starts = ((0, 0, 2), (0, 0, -2))
    ends = ((0, 0, -2), (0, 0, 2))
    ours, theirs = clients
    got = mojo.rayTestBatch(starts, ends, physicsClientId=ours)
    expected = bullet.rayTestBatch(starts, ends, physicsClientId=theirs)
    for a, b in zip(got, expected):
        assert a[0:3] == pytest.approx(b[0:3])
        assert_vec(a[3], b[3])
        assert_vec(a[4], b[4])


def test_ray_starting_inside_convex_shape_reports_no_hit(clients):
    box = make_shape_pair(clients, mojo.GEOM_BOX, halfExtents=(1, 1, 1))
    make_body_pair(clients, box)
    ours, theirs = clients
    got = mojo.rayTest((0, 0, 0), (2, 0, 0), physicsClientId=ours)
    expected = bullet.rayTest((0, 0, 0), (2, 0, 0), physicsClientId=theirs)
    assert got == expected


def test_transform_helpers_match_pybullet():
    euler = (0.3, -0.7, 1.2)
    q = mojo.getQuaternionFromEuler(euler)
    assert_vec(q, bullet.getQuaternionFromEuler(euler))
    assert_vec(mojo.getEulerFromQuaternion(q), bullet.getEulerFromQuaternion(q))
    assert_vec(mojo.getMatrixFromQuaternion(q), bullet.getMatrixFromQuaternion(q))
    args = ((1, 2, 3), q, (-0.5, 0.4, 1.2), mojo.getQuaternionFromEuler((-0.2, 0.1, 0.6)))
    got = mojo.multiplyTransforms(*args)
    expected = bullet.multiplyTransforms(*args)
    assert_vec(got[0], expected[0], 4e-7)
    assert_vec(got[1], expected[1], 2e-7)
    got = mojo.invertTransform(args[0], args[1])
    expected = bullet.invertTransform(args[0], args[1])
    assert_vec(got[0], expected[0], 4e-7)
    assert_vec(got[1], expected[1], 2e-7)


def test_free_fall_tracks_pybullet(clients):
    shapes = make_shape_pair(clients, mojo.GEOM_SPHERE, radius=0.4)
    bodies = make_body_pair(clients, shapes, mass=2.0, position=(0, 0, 5))
    ours, theirs = clients
    for module, client in ((mojo, ours), (bullet, theirs)):
        module.setGravity(0, 0, -10, physicsClientId=client)
        module.setTimeStep(0.002, physicsClientId=client)
    for _ in range(200):
        mojo.stepSimulation(ours)
        bullet.stepSimulation(theirs)
    got_pos, _ = mojo.getBasePositionAndOrientation(bodies[0], ours)
    expected_pos, _ = bullet.getBasePositionAndOrientation(bodies[1], theirs)
    got_vel, _ = mojo.getBaseVelocity(bodies[0], ours)
    expected_vel, _ = bullet.getBaseVelocity(bodies[1], theirs)
    assert_vec(got_pos, expected_pos, 3e-4)
    assert_vec(got_vel, expected_vel, 2e-3)


def test_external_force_at_center_matches_linear_velocity(clients):
    shapes = make_shape_pair(clients, mojo.GEOM_BOX, halfExtents=(0.5, 0.75, 1))
    bodies = make_body_pair(clients, shapes, mass=4.0, position=(0, 0, 2))
    ours, theirs = clients
    for module, client, body in ((mojo, ours, bodies[0]), (bullet, theirs, bodies[1])):
        module.setTimeStep(0.01, physicsClientId=client)
        module.changeDynamics(
            body, -1, linearDamping=0, angularDamping=0, physicsClientId=client
        )
        module.applyExternalForce(
            body, -1, (8, -4, 2), (0, 0, 2), module.WORLD_FRAME, physicsClientId=client
        )
        module.stepSimulation(physicsClientId=client)
    got = mojo.getBaseVelocity(bodies[0], ours)
    expected = bullet.getBaseVelocity(bodies[1], theirs)
    assert_vec(got[0], expected[0], 1e-12)
    assert_vec(got[1], expected[1], 1e-12)


def test_step_clears_force_and_torque_simd_tail():
    client = mojo.connect(mojo.DIRECT)
    try:
        shape = mojo.createCollisionShape(
            mojo.GEOM_SPHERE, radius=0.1, physicsClientId=client
        )
        for i in range(3):
            body = mojo.createMultiBody(
                1,
                shape,
                basePosition=(2.0 * i, 0, 1),
                physicsClientId=client,
            )
            mojo.applyExternalForce(
                body,
                -1,
                (i + 1, i + 2, i + 3),
                (2.0 * i, 0, 1),
                mojo.WORLD_FRAME,
                physicsClientId=client,
            )
            mojo.applyExternalTorque(
                body,
                -1,
                (i + 4, i + 5, i + 6),
                mojo.WORLD_FRAME,
                physicsClientId=client,
            )
        mojo.stepSimulation(client)
        world = mojo._world(client)
        assert np.count_nonzero(world.force) == 0
        assert np.count_nonzero(world.torque) == 0
    finally:
        mojo.disconnect(client)


@pytest.mark.parametrize("body_count", [1023, 1024])
def test_large_independent_integration_parallel_threshold(body_count):
    client = mojo.connect(mojo.DIRECT)
    try:
        shape = mojo.createCollisionShape(
            mojo.GEOM_SPHERE, radius=0.1, physicsClientId=client
        )
        for i in range(body_count):
            mojo.createMultiBody(
                1,
                shape,
                basePosition=(i * 2.0, 0, 10),
                physicsClientId=client,
            )
        mojo.setGravity(0, 0, -10, physicsClientId=client)
        mojo.setTimeStep(0.01, physicsClientId=client)
        mojo.stepSimulation(client)
        world = mojo._world(client)
        assert world.vel[:, 2] == pytest.approx(np.full(body_count, -0.1), abs=1e-12)
        assert world.pos[:, 2] == pytest.approx(np.full(body_count, 9.999), abs=1e-12)
    finally:
        mojo.disconnect(client)


def test_sphere_settles_on_plane_like_pybullet(clients):
    plane = make_shape_pair(clients, mojo.GEOM_PLANE, planeNormal=(0, 0, 1))
    sphere = make_shape_pair(clients, mojo.GEOM_SPHERE, radius=0.5)
    make_body_pair(clients, plane)
    bodies = make_body_pair(clients, sphere, mass=1, position=(0, 0, 2))
    ours, theirs = clients
    for module, client in ((mojo, ours), (bullet, theirs)):
        module.setGravity(0, 0, -9.8, physicsClientId=client)
        module.setTimeStep(1 / 240, physicsClientId=client)
    for _ in range(480):
        mojo.stepSimulation(ours)
        bullet.stepSimulation(theirs)
    got_pos, _ = mojo.getBasePositionAndOrientation(bodies[0], ours)
    expected_pos, _ = bullet.getBasePositionAndOrientation(bodies[1], theirs)
    got_vel, _ = mojo.getBaseVelocity(bodies[0], ours)
    assert got_pos[2] == pytest.approx(expected_pos[2], abs=2e-4)
    assert got_vel[2] == pytest.approx(0.0, abs=1e-10)


def test_dynamics_info_and_change_dynamics(clients):
    shape = make_shape_pair(clients, mojo.GEOM_SPHERE, radius=0.5)
    body = make_body_pair(clients, shape, mass=2)
    ours, theirs = clients
    for module, client, body_id in ((mojo, ours, body[0]), (bullet, theirs, body[1])):
        module.changeDynamics(
            body_id,
            -1,
            lateralFriction=0.8,
            restitution=0.3,
            linearDamping=0.02,
            angularDamping=0.03,
            physicsClientId=client,
        )
    got = mojo.getDynamicsInfo(body[0], -1, ours)
    expected = bullet.getDynamicsInfo(body[1], -1, theirs)
    assert got[0] == expected[0]
    assert got[1] == expected[1]
    assert_vec(got[2], expected[2])
    assert got[5] == expected[5]


def test_overlapping_objects_matches_expected_ids(clients):
    shapes = make_shape_pair(clients, mojo.GEOM_SPHERE, radius=0.5)
    bodies = [
        make_body_pair(clients, shapes, position=(x, 0, 0))
        for x in (-2.0, 0.0, 2.0)
    ]
    ours, _ = clients
    got = mojo.getOverlappingObjects((-0.6, -1, -1), (2.6, 1, 1), ours)
    assert got == ((bodies[1][0], -1), (bodies[2][0], -1))


def test_unsupported_geometry_fails_explicitly(clients):
    ours, _ = clients
    with pytest.raises(NotImplementedError):
        mojo.createCollisionShape(5, physicsClientId=ours)
