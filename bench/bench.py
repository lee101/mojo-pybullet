"""Measured collision and dynamics comparisons against PyBullet."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np
import pybullet as bullet

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python")
)
import mojo_pybullet as mojo  # noqa: E402


def timeit(function, repeat=5):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def setup(module):
    client = module.connect(module.DIRECT)
    sphere = module.createCollisionShape(
        module.GEOM_SPHERE, radius=0.3, physicsClientId=client
    )
    box = module.createCollisionShape(
        module.GEOM_BOX, halfExtents=(0.3, 0.4, 0.5), physicsClientId=client
    )
    for i in range(64):
        x = (i % 8 - 3.5) * 0.9
        y = (i // 8 - 3.5) * 0.9
        shape = sphere if i % 2 == 0 else box
        orientation = module.getQuaternionFromEuler((0.1 * i, -0.03 * i, 0.07 * i))
        module.createMultiBody(
            0,
            shape,
            -1,
            (x, y, 0),
            orientation,
            physicsClientId=client,
        )
    return client, sphere


def ray_case(ours, theirs, device="cpu"):
    rng = np.random.default_rng(7)
    starts = np.column_stack(
        (
            rng.uniform(-4, 4, 8_192),
            rng.uniform(-4, 4, 8_192),
            np.full(8_192, 3.0),
        )
    )
    ends = starts.copy()
    ends[:, 2] = -3.0
    return (
        lambda: mojo.rayTestBatch(
            starts, ends, physicsClientId=ours, device=device
        ),
        lambda: bullet.rayTestBatch(starts, ends, physicsClientId=theirs),
    )


def closest_case(ours, theirs):
    return (
        lambda: [
            mojo.getClosestPoints(
                2 * (i % 32), 2 * ((i * 17 + 1) % 32), 10, physicsClientId=ours
            )
            for i in range(20_000)
        ],
        lambda: [
            bullet.getClosestPoints(
                2 * (i % 32), 2 * ((i * 17 + 1) % 32), 10, physicsClientId=theirs
            )
            for i in range(20_000)
        ],
    )


def dynamics_setup(module):
    client = module.connect(module.DIRECT)
    shape = module.createCollisionShape(
        module.GEOM_SPHERE, radius=0.2, physicsClientId=client
    )
    for i in range(128):
        module.createMultiBody(
            1,
            shape,
            -1,
            ((i % 16) * 2.0, (i // 16) * 2.0, 100 + i * 0.01),
            physicsClientId=client,
        )
    module.setGravity(0, 0, -9.8, physicsClientId=client)
    return client


def main():
    ours, _ = setup(mojo)
    theirs, _ = setup(bullet)
    dyn_ours = dynamics_setup(mojo)
    dyn_theirs = dynamics_setup(bullet)
    cases = [
        ("rayTestBatch, 8,192 rays x 64 bodies", *ray_case(ours, theirs)),
        ("getClosestPoints, 20,000 calls", *closest_case(ours, theirs)),
        (
            "stepSimulation, 128 bodies x 100 steps",
            lambda: [mojo.stepSimulation(dyn_ours) for _ in range(100)],
            lambda: [bullet.stepSimulation(dyn_theirs) for _ in range(100)],
        ),
    ]
    if mojo._gpu_memory_ready():
        cases.insert(
            1,
            (
                "rayTestBatch GPU, 8,192 rays x 64 bodies",
                *ray_case(ours, theirs, device="gpu"),
            ),
        )
    else:
        print("GPU benchmark skipped: less than 4000 MiB free or no GPU available")
    print(f"Machine: {platform.processor() or platform.machine()}, {platform.system()} {platform.release()}")
    print("| Case | mojo-pybullet | PyBullet | Relative |")
    print("|---|---:|---:|---:|")
    for name, ours_fn, theirs_fn in cases:
        ours_result = ours_fn()
        theirs_result = theirs_fn()
        if name.startswith("rayTestBatch"):
            assert len(ours_result) == len(theirs_result) == 8_192
        ours_time = timeit(ours_fn)
        theirs_time = timeit(theirs_fn)
        speed = theirs_time / ours_time
        label = f"{speed:.2f}x faster" if speed >= 1 else f"{1 / speed:.2f}x slower"
        print(
            f"| {name} | {ours_time * 1000:.2f} ms | "
            f"{theirs_time * 1000:.2f} ms | {label} |"
        )
    mojo.disconnect(ours)
    mojo.disconnect(dyn_ours)
    bullet.disconnect(theirs)
    bullet.disconnect(dyn_theirs)


if __name__ == "__main__":
    main()
