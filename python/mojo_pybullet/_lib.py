"""ctypes bindings for the compiled Mojo kernels."""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_PYBULLET_LIB", os.path.join(ROOT, "dist", "libmojo-pybullet.so"))
I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mpb_closest_points": ([I, I, I, I, I, I, I], I),
    "mpb_ray_test_batch": ([I] * 14, None),
    "mpb_ray_test_batch_gpu": ([I] * 14, I),
    "mpb_step": ([I] * 16 + [F, F, F, F, I, I], None),
}


def build() -> str:
    if not os.path.exists(LIB):
        proc = subprocess.run(
            ["bash", os.path.join(ROOT, "build", "build.sh")],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if proc.returncode:
            raise RuntimeError((proc.stderr or proc.stdout).strip())
    return LIB


_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def addr(array: np.ndarray) -> int:
    return int(array.ctypes.data)
