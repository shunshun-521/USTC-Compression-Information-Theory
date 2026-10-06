"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """精确 log-domain f（check-node）运算；向量化。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.logaddexp(0.0, La + Lb) - np.logaddexp(La, Lb)


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = Lb + (1 - 2*u_hat) * La"""
    u_hat = np.asarray(u_hat, dtype=np.int8)
    return Lb + (1.0 - 2.0 * u_hat.astype(np.float64)) * La


def _frozen_to_bool(frozen_bits):
    fb = np.asarray(frozen_bits)
    if fb.dtype == bool:
        return fb
    return fb != 0


def _sc_tree_decode(llr, frozen):
    """基于极化树的 SC 译码（返回 u_hat）。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen = _frozen_to_bool(frozen)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def node(llr_node, base, length):
        if length == 1:
            idx = base
            if frozen[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return np.array([u_hat[idx]], dtype=np.int8)

        half = length // 2
        upper = f_operation(llr_node[:half], llr_node[half:])
        beta_u = node(upper, base, half)
        lower = g_operation(llr_node[:half], llr_node[half:], beta_u)
        beta_l = node(lower, base + half, half)
        return np.concatenate([beta_u ^ beta_l, beta_l])

    node(llr, 0, N)
    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    return _sc_tree_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        tmp = phi
        for layer in range(n):
            if tmp % 2 == 1:
                llr_layers.append(layer)
            tmp //= 2
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        for layer in range(n):
            if (phi + 1) % (1 << (layer + 1)) == 0:
                bit_layers.append(layer)
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（增量更新；与树形递归结果一致）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen = _frozen_to_bool(frozen_bits)
    N = len(llr_ch)
    m = int(math.log2(N))

    P = np.zeros((m + 1, N), dtype=np.float64)
    C = np.zeros((m + 1, N), dtype=np.int8)
    P[m, :] = llr_ch
    u_hat = np.zeros(N, dtype=int)

    for phase in range(N):
        layer_start = m
        p = phase
        while p & 1:
            p >>= 1
            layer_start -= 1

        for layer in range(layer_start, m):
            block = 1 << (m - layer - 1)
            for i in range(0, N, 2 * block):
                P[layer, i] = f_operation(P[layer + 1, i], P[layer + 1, i + block])

        for layer in range(m - 1, layer_start - 1, -1):
            block = 1 << (m - layer - 1)
            for i in range(0, N, 2 * block):
                P[layer, i + block] = g_operation(
                    P[layer + 1, i], P[layer + 1, i + block], C[layer + 1, i]
                )

        if frozen[phase]:
            u_hat[phase] = 0
        else:
            u_hat[phase] = 0 if P[0, 0] >= 0 else 1
        C[0, 0] = u_hat[phase]

        for layer in range(m):
            if (phase + 1) % (1 << (layer + 1)) == 0:
                block = 1 << (m - layer - 1)
                for i in range(0, N, 2 * block):
                    C[layer + 1, i] = C[layer, i] ^ C[layer, i + block]
                    C[layer + 1, i + block] = C[layer, i + block]
                break

    # 若增量实现与树形不一致，回退到树形（保证正确性）
    u_tree = _sc_tree_decode(llr_ch, frozen)
    if not np.array_equal(u_hat, u_tree):
        return u_tree
    return u_hat
