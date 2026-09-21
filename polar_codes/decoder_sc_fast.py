"""Numba 加速的 SC 译码核心（可选）。"""
import numpy as np

try:
    from numba import njit

    @njit
    def _f_op(a, b):
        sa = 1.0 if a >= 0 else -1.0
        sb = 1.0 if b >= 0 else -1.0
        return sa * sb * min(abs(a), abs(b))

    @njit
    def _g_op(a, b, u):
        return (1.0 - 2.0 * u) * a + b

    @njit
    def _sc_core(llr, frozen):
        n = llr.shape[0]
        if n == 1:
            bit = 0
            if not frozen[0]:
                bit = 0 if llr[0] >= 0 else 1
            u = np.empty(1, dtype=np.int8)
            u[0] = bit
            up = np.empty(1, dtype=np.int8)
            up[0] = bit
            return u, up

        h = n // 2
        llr_l = np.empty(h, dtype=np.float64)
        for i in range(h):
            llr_l[i] = _f_op(llr[i], llr[i + h])
        u_l, up_l = _sc_core(llr_l, frozen[:h])
        llr_r = np.empty(h, dtype=np.float64)
        for i in range(h):
            llr_r[i] = _g_op(llr[i], llr[i + h], up_l[i])
        u_r, up_r = _sc_core(llr_r, frozen[h:])

        u = np.empty(n, dtype=np.int8)
        up = np.empty(n, dtype=np.int8)
        for i in range(h):
            u[i] = u_l[i]
            u[i + h] = u_r[i]
            up[i] = up_l[i] ^ up_r[i]
            up[i + h] = up_r[i]
        return u, up

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


def sc_decode_numba(llr, frozen_bits):
    """若可用 Numba，则使用 JIT 加速 SC 译码。"""
    if not HAS_NUMBA:
        return None
    llr = np.asarray(llr, dtype=np.float64)
    frozen = np.asarray(frozen_bits, dtype=np.bool_)
    u_hat, _ = _sc_core(llr, frozen)
    return u_hat.astype(int)
