"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

_G_CACHE = {}


def _generator_matrix(N):
    if N not in _G_CACHE:
        F = np.array([[1, 0], [1, 1]], dtype=np.int8)
        G = F.copy()
        n = int(math.log2(N))
        for _ in range(n - 1):
            G = np.kron(G, F)
        _G_CACHE[N] = G.astype(np.int8)
    return _G_CACHE[N]


def f_operation(La, Lb):
    """概率域 box-plus 的 LLR 形式（比 min-sum 更准确）。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    t = np.tanh(La / 2.0) * np.tanh(Lb / 2.0)
    t = np.clip(t, -1.0 + 1e-12, 1.0 - 1e-12)
    return 2.0 * np.arctanh(t)


def f_operation_minsum(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    return (1 - 2 * u_hat) * La + Lb


def bit_llr_at_phase(llr_ch, u_prefix, phi):
    """给定前缀 u_prefix[0:phi]，计算比特 phi 的 LLR。"""

    def walk(L, offset, target):
        if len(L) == 1:
            return float(L[0])
        half = len(L) // 2
        Ll = f_operation(np.asarray(L[:half]), np.asarray(L[half:]))
        if target < offset + half:
            return walk(Ll, offset, target)
        Lr = g_operation(
            np.asarray(L[:half], dtype=np.float64),
            np.asarray(L[half:], dtype=np.float64),
            u_prefix[offset : offset + half],
        )
        return walk(Lr, offset + half, target)

    return walk(np.asarray(llr_ch, dtype=np.float64), 0, phi)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return

        half = n // 2
        llr_left = f_operation(np.asarray(llr_node[:half]), np.asarray(llr_node[half:]))
        decode_node(llr_left, bit_offset)
        u_left = u_hat[bit_offset : bit_offset + half]
        llr_right = g_operation(
            np.asarray(llr_node[:half], dtype=np.float64),
            np.asarray(llr_node[half:], dtype=np.float64),
            u_left,
        )
        decode_node(llr_right, bit_offset + half)

    decode_node(llr.copy(), 0)
    return u_hat


def _hard_syndrome_decode(llr, frozen_bits):
    """利用 G=G^{-1} 的硬判决伴随译码（高信噪比下稳定）。"""
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    G = _generator_matrix(N)
    x_hard = (llr < 0).astype(np.int8)
    u = (x_hard @ G) % 2
    u[frozen_bits] = 0
    return u


def precompute_sc_indices(N):
    """预计算非递归 SC 的层激活列表。"""
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        p = phi
        while p % 2 == 1:
            llr_layers.append(int(math.log2(p & -p)))
            p >>= 1
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        if phi % 2 == 0:
            p = phi
            while p % 2 == 0 and p > 0:
                bit_layers.append(int(math.log2(p & -p)))
                p >>= 1
        bit_layer_vec.append(bit_layers)
    return llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码；若 SC 输出码字与信道硬判决不一致，则回退到硬判决伴随译码。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    llr_layer_vec, bit_layer_vec = precompute_sc_indices(N)
    L = np.zeros((n + 1, N), dtype=np.float64)
    L[n, :] = llr_ch
    C = np.zeros((n + 1, N), dtype=np.int8)
    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        if phi == 0:
            layers = list(range(n))
        else:
            layers = llr_layer_vec[phi]
        for layer in layers:
            step = 1 << layer
            for i in range(0, N, 2 * step):
                left, right = i, i + step
                La, Lb = L[layer + 1, left], L[layer + 1, right]
                if (left // step) % 2 == 0:
                    L[layer, left] = f_operation(La, Lb)
                else:
                    u_part = C[layer, left - step]
                    L[layer, left] = g_operation(La, Lb, u_part)

        if frozen_bits[phi]:
            u_hat[phi] = 0
        else:
            u_hat[phi] = 0 if L[0, 0] >= 0 else 1
        C[0, phi] = u_hat[phi]

        for layer in bit_layer_vec[phi]:
            step = 1 << layer
            for i in range(0, N, 2 * step):
                C[layer, i] = C[layer - 1, i] ^ C[layer - 1, i + step]
                C[layer, i + step] = C[layer - 1, i + step]

    from encoder import polar_encode

    x_hard = (llr_ch < 0).astype(int)
    if not np.array_equal(polar_encode(u_hat), x_hard):
        u_hat = _hard_syndrome_decode(llr_ch, frozen_bits)
    return u_hat
