"""
极化码 SC（串行抵消）译码器
与 encoder / BP 因子图一致的 min-sum SC 实现
"""
import math
import numpy as np


def f_operation(La, Lb):
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    return (1 - 2 * u_hat) * La + Lb


def _minsum(a, b, alpha=1.0):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（与主译码相同逻辑）"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    n = int(math.log2(N))
    lambda_offset = [0] * (n + 1)
    for layer in range(1, n + 1):
        lambda_offset[layer] = lambda_offset[layer - 1] + (1 << (n - layer + 1)) - 1

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers_llr = []
        p = phi
        while p & 1:
            layers_llr.append(int(math.log2(p & -p)))
            p >>= 1
        llr_layer_vec.append(layers_llr)

        if phi % 2 == 0:
            layers_bit = list(range(n))
        else:
            p = phi
            while p & 1:
                p >>= 1
            layers_bit = list(range(int(math.log2(p)) if p else 0))
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码。
    frozen_bits: 1 表示冻结位
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    large = 1e8

    L = np.zeros((N, n + 1), dtype=np.float64)
    R = np.zeros((N, n + 1), dtype=np.float64)
    L[:, n] = llr_ch
    R[frozen_bits, 0] = large

    u_hat = np.zeros(N, dtype=np.int8)

    for phi in range(N):
        for j in range(n, 0, -1):
            s = 1 << (j - 1)
            for i in range(0, N, s << 1):
                for k in range(s):
                    idx = i + k
                    idx2 = idx + s
                    L[idx, j - 1] = _minsum(R[idx, j] + L[idx2, j], L[idx, j])
                    L[idx2, j - 1] = _minsum(R[idx, j], L[idx, j]) + L[idx2, j]

        total = L[phi, 0] + R[phi, 0]
        if frozen_bits[phi]:
            u_hat[phi] = 0
        else:
            u_hat[phi] = 0 if total >= 0 else 1

        R[phi, 0] = large if u_hat[phi] == 0 else -large

    return u_hat.astype(int)
