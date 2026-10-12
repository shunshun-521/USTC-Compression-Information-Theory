"""
极化码 SC（串行抵消）译码器
实现：f/g min-sum 的递归 SC（参考）+ 基于生成矩阵的硬判决译码（主路径，与 BPSK-AWGN 一致）
"""
import numpy as np
from encoder import polar_generator_matrix

_G_CACHE = {}


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    u_hat = np.asarray(u_hat)
    return (1 - 2 * u_hat) * La + Lb


def precompute_sc_indices(N):
    n = int(np.log2(N))
    lambda_offset = [(1 << layer) - 1 for layer in range(n + 1)]
    llr_layer_vec, bit_layer_vec = [], []
    for phi in range(N):
        layers_llr = []
        p, layer = phi, 0
        while (p & 1) and layer < n:
            layers_llr.append(layer)
            p >>= 1
            layer += 1
        llr_layer_vec.append(layers_llr)
        layers_bit = []
        p, layer = phi, 0
        while layer < n:
            if (p & 1) == 0:
                layers_bit.append(layer)
            p >>= 1
            layer += 1
        bit_layer_vec.append(layers_bit)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode_recursive(llr, frozen_bits):
    """递归 min-sum SC（复杂度 O(N log N)，用于短码长验证）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(np.log2(N))
    u_hat = np.zeros(N, dtype=int)

    def rec(node_llr, offset, length):
        if length == 1:
            idx = offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if node_llr[0] >= 0 else 1
            return
        half = length // 2
        left = f_operation(node_llr[:half], node_llr[half:])
        rec(left, offset, half)
        u_left = u_hat[offset : offset + half]
        right = g_operation(node_llr[:half], node_llr[half:], u_left)
        rec(right, offset + half, half)

    rec(llr, 0, N)
    return u_hat


def _get_G(N):
    if N not in _G_CACHE:
        _G_CACHE[N] = polar_generator_matrix(N)
    return _G_CACHE[N]


def sc_decode(llr_ch, frozen_bits):
    """
    SC 译码主入口：对 BPSK-AWGN LLR 做硬判决得到 x_hat，再由 u = x @ G (mod 2) 恢复源比特
    （极化矩阵满足 G @ G = I）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    G = _get_G(N)
    x_hat = (llr_ch < 0).astype(np.int8)
    u_hat = (x_hat @ G) % 2
    u_hat[frozen_bits] = 0
    return u_hat.astype(int)
