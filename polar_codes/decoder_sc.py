"""
极化码 SC（串行抵消）译码器
"""
import numpy as np
import math
from encoder import prepare_channel_llr


def f_operation(La, Lb):
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    u_hat = np.asarray(u_hat)
    return (1 - 2 * u_hat) * La + Lb


def _f_min_sum(a, b, alpha=1.0):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


def _sc_successive_cancellation(llr_br, frozen_bits, alpha=1.0):
    """基于因子图逐比特更新的 SC 译码。"""
    llr_br = np.asarray(llr_br, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_br)
    n = int(math.log2(N))
    large = 1e6

    u_hat = np.zeros(N, dtype=int)
    L = np.zeros((N, n + 1), dtype=np.float64)
    R = np.zeros((N, n + 1), dtype=np.float64)

    for phi in range(N):
        L.fill(0.0)
        R.fill(0.0)
        L[:, n] = llr_br
        R[frozen_bits, 0] = large
        for i in range(phi):
            if not frozen_bits[i]:
                R[i, 0] = large if u_hat[i] == 0 else -large

        for j in range(n, 0, -1):
            s = 1 << (j - 1)
            for block in range(0, N, 2 * s):
                for i in range(block, block + s):
                    ip = i + s
                    L[i, j - 1] = _f_min_sum(R[i, j] + L[ip, j], L[i, j], alpha)
                    L[ip, j - 1] = _f_min_sum(R[i, j], L[i, j], alpha) + L[ip, j]

        if frozen_bits[phi]:
            u_hat[phi] = 0
        else:
            u_hat[phi] = 0 if (L[phi, 0] + R[phi, 0]) >= 0 else 1

    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        psi = phi
        lam = 0
        while psi & 1:
            lam += 1
            psi >>= 1
        llr_layer_vec.append(list(range(lam, n)))
        bit_layers = []
        psi = phi
        while psi & 1:
            bit_layers.append(int(math.log2(psi & -psi)))
            psi >>= 1
        bit_layer_vec.append(bit_layers)
    return None, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    llr_br = prepare_channel_llr(llr_ch)
    return _sc_successive_cancellation(llr_br, frozen_bits)
