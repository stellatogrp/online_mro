"""ctypes wrapper for the C implementation of ``worst_case_value_box``.

The shared library ``libworstcase.{dylib,so}`` is built next to this file by
``build.sh``.  ``worst_case_value_box_c`` has the same signature and return
convention as the numpy ``portfolio.utils.worst_case_value_box`` (which stays
untouched as the verification oracle); ``inner_mode`` selects the per-sample
inner solver (0 = literal 80-iteration mu-bisection, numpy-parity path;
1 = exact piecewise closed form of the same optimality condition, default).
"""
import ctypes
import os
import sys

import numpy as np

_LIBNAME = "libworstcase.dylib" if sys.platform == "darwin" else "libworstcase.so"
_LIBPATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), _LIBNAME)

_LIB = None


def _load():
    """Load (and cache) the shared library; None if absent/unloadable.

    Retried whenever the library file exists, so building after import works.
    """
    global _LIB
    if _LIB is not None:
        return _LIB
    if not os.path.exists(_LIBPATH):
        return None
    try:
        lib = ctypes.CDLL(_LIBPATH)
    except OSError:
        return None
    dp = ctypes.POINTER(ctypes.c_double)
    up = ctypes.POINTER(ctypes.c_ubyte)
    lib.worstcase_box.restype = ctypes.c_int
    lib.worstcase_box.argtypes = [
        dp, ctypes.c_long, ctypes.c_int,            # dat, N, d
        dp, ctypes.c_double, dp,                    # x, tau, w
        ctypes.c_double, dp, dp, ctypes.c_double,   # eps, lb, ub, a
        ctypes.c_int, ctypes.c_int, ctypes.c_int,   # lam_iter, mu_iter, inner_mode
        dp, dp,                                     # F_out, lam_out
        dp, up,                                     # zeta_out, active_out (nullable)
    ]
    _LIB = lib
    return _LIB


def available():
    """True iff the compiled library is present and loads."""
    return _load() is not None


def _as_c(arr, shape=None):
    a = np.ascontiguousarray(arr, dtype=np.float64)
    if shape is not None and a.shape != shape:
        a = np.ascontiguousarray(np.broadcast_to(a, shape), dtype=np.float64)
    return a


def worst_case_value_box_c(x, tau, dat, w, eps, lb, ub, a=-5.0,
                           lam_iter=60, mu_iter=80, return_state=False,
                           inner_mode=1):
    """C version of ``worst_case_value_box`` (same signature/returns).

    Returns ``F`` or, with ``return_state``, ``(F, lam_star, zeta, active)``.
    """
    lib = _load()
    if lib is None:
        raise RuntimeError(
            "libworstcase not available; build it with portfolio/cworst/build.sh")
    dat = _as_c(dat)
    N, d = dat.shape
    x = _as_c(x, (d,))
    w = _as_c(w, (N,))
    lb = _as_c(lb, (d,))
    ub = _as_c(ub, (d,))

    dp = ctypes.POINTER(ctypes.c_double)
    up = ctypes.POINTER(ctypes.c_ubyte)
    F = ctypes.c_double()
    lam = ctypes.c_double()
    if return_state:
        zeta = np.empty((N, d), dtype=np.float64)
        active = np.empty(N, dtype=np.uint8)
        zeta_p = zeta.ctypes.data_as(dp)
        active_p = active.ctypes.data_as(up)
    else:
        zeta = active = None
        zeta_p = ctypes.cast(None, dp)
        active_p = ctypes.cast(None, up)

    ret = lib.worstcase_box(
        dat.ctypes.data_as(dp), ctypes.c_long(N), ctypes.c_int(d),
        x.ctypes.data_as(dp), ctypes.c_double(float(tau)),
        w.ctypes.data_as(dp),
        ctypes.c_double(float(eps)), lb.ctypes.data_as(dp),
        ub.ctypes.data_as(dp), ctypes.c_double(float(a)),
        ctypes.c_int(lam_iter), ctypes.c_int(mu_iter), ctypes.c_int(inner_mode),
        ctypes.byref(F), ctypes.byref(lam), zeta_p, active_p)
    if ret != 0:
        raise RuntimeError("worstcase_box failed (allocation error)")
    if return_state:
        return float(F.value), float(lam.value), zeta, active.astype(bool)
    return float(F.value)
