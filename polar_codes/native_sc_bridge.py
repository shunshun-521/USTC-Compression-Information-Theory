import ctypes
import os
import numpy as np

_LIB = None


def _load_lib():
    global _LIB
    if _LIB is not None:
        return _LIB
    lib_path = os.path.join(os.path.dirname(__file__), "native_sc.so")
    if not os.path.exists(lib_path):
        raise RuntimeError("native_sc.so not found; run compile_native.sh")
    _LIB = ctypes.CDLL(lib_path)
    _LIB.polar_sc_decode.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_uint8,
    ]
    _LIB.polar_sc_decode.restype = None
    return _LIB


def native_sc_decode(llr, frozen_bits):
    llr = np.ascontiguousarray(llr, dtype=np.float64)
    frozen = np.ascontiguousarray(frozen_bits.astype(np.uint8), dtype=np.uint8)
    n = int(np.log2(len(llr)))
    out = np.zeros(len(llr), dtype=np.uint8)
    lib = _load_lib()
    lib.polar_sc_decode(
        llr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        frozen.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_uint8(n),
    )
    return out.astype(int)
