"""Numba 加速的 SC 译码（可选）"""
import numpy as np

try:
    from numba import njit

    @njit
    def _f_op(La, Lb):
        sa = 1.0 if La >= 0 else -1.0
        sb = 1.0 if Lb >= 0 else -1.0
        return sa * sb * min(abs(La), abs(Lb))

    @njit
    def _g_op(La, Lb, u):
        return (1.0 - 2.0 * u) * La + Lb

    @njit
    def sc_decode_numba(llr_ch, frozen_bits, br):
        N = len(llr_ch)
        n = 0
        while (1 << n) < N:
            n += 1
        L = np.zeros((n + 1, N))
        C = np.zeros((n + 1, N), dtype=np.int64)
        for i in range(N):
            L[n, i] = llr_ch[br[i]]
        u_hat = np.zeros(N, dtype=np.int64)
        for phi in range(N):
            for layer in range(n - 1, -1, -1):
                step = 1 << layer
                i = 0
                while i < N:
                    j = i
                    while j < i + step:
                        L[layer, j] = _f_op(L[layer + 1, j], L[layer + 1, j + step])
                        L[layer, j + step] = _g_op(
                            L[layer + 1, j], L[layer + 1, j + step], C[layer, j]
                        )
                        j += 1
                    i += 2 * step
            if frozen_bits[phi] == 1:
                u_hat[phi] = 0
            else:
                u_hat[phi] = 0 if L[0, phi] >= 0 else 1
            C[0, phi] = u_hat[phi]
            for layer in range(n):
                step = 1 << layer
                i = 0
                while i < N:
                    j = i
                    while j < i + step:
                        C[layer + 1, j] = C[layer, j] ^ C[layer, j + step]
                        C[layer + 1, j + step] = C[layer, j + step]
                        j += 1
                    i += 2 * step
        return u_hat

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def sc_decode_numba(llr_ch, frozen_bits, br):
        raise RuntimeError("numba not available")
