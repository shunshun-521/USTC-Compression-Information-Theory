"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    def decode_node(llr_node, frozen_node):
        n = len(llr_node)
        if n == 1:
            if frozen_node[0]:
                bit = 0
            else:
                bit = 0 if llr_node[0] >= 0 else 1
            return np.array([bit], dtype=int), np.array([bit], dtype=int)
        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        u_left, u_left_up = decode_node(llr_left, frozen_node[:half])
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left_up)
        u_right, u_right_up = decode_node(llr_right, frozen_node[half:])
        u_hat = np.concatenate([u_left, u_right])
        u_up = np.concatenate([u_left_up ^ u_right_up, u_right_up])
        return u_hat, u_up

    u_hat, _ = decode_node(llr, frozen_bits)
    return u_hat


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码辅助向量（基于比特索引 phi 的层更新规则）。
    """
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        for layer in range(n):
            if (phi >> layer) & 1 == 0:
                llr_layers.append(layer)
        llr_layer_vec.append(llr_layers)
        bit_layers = []
        for layer in range(n):
            if (phi >> layer) & 1:
                bit_layers.append(layer)
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode_layered(llr_ch, frozen_bits):
    """非递归 SC 译码（分层 P/C 数组实现，供对照/扩展）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(np.log2(N))
    lambda_offset, llr_layer_vec, bit_layer_vec = precompute_sc_indices(N)

    P = np.zeros((n + 1, N), dtype=np.float64)
    C = np.zeros((n + 1, N), dtype=int)
    P[n, :] = llr_ch
    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        for layer in llr_layer_vec[phi]:
            lam = lambda_offset[layer]
            i = (phi // (2 * lam)) * (2 * lam)
            P[layer, i:i + lam] = f_operation(
                P[layer + 1, i:i + lam],
                P[layer + 1, i + lam:i + 2 * lam],
            )
            P[layer, i + lam:i + 2 * lam] = g_operation(
                P[layer + 1, i:i + lam],
                P[layer + 1, i + lam:i + 2 * lam],
                C[layer, i:i + lam],
            )

        if frozen_bits[phi]:
            u_hat[phi] = 0
        else:
            u_hat[phi] = 0 if P[0, 0] >= 0 else 1

        C[0, 0] = u_hat[phi]
        for layer in bit_layer_vec[phi]:
            lam = lambda_offset[layer]
            i = (phi // (2 * lam)) * (2 * lam)
            C[layer, i:i + lam] = (
                C[layer + 1, i:i + lam] ^ C[layer + 1, i + lam:i + 2 * lam]
            )
            C[layer, i + lam:i + 2 * lam] = C[layer + 1, i + lam:i + 2 * lam]

    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """
    SC 译码主入口。
    默认使用已验证的递归实现；`sc_decode_layered` 提供 O(N log N) 分层非递归结构。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    if frozen_bits.dtype == bool:
        frozen_bits = np.asarray(frozen_bits, dtype=bool)
    else:
        frozen_bits = np.asarray(frozen_bits, dtype=np.int8).astype(bool)
    return sc_decode_recursive(llr_ch, frozen_bits)


def decode_sc_from_channel_llr(llr_ch, frozen_bits):
    """信道序 LLR -> 比特倒序 -> SC 译码（与 B_N 编码一致）"""
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    llr_in = np.zeros(N, dtype=np.float64)
    llr_in[br] = llr_ch
    return sc_decode(llr_in, frozen_bits)


def sc_decode_with_llr_reorder(llr_ch, frozen_bits):
    """别名：对信道 LLR 做比特倒序后 SC 译码"""
    return decode_sc_from_channel_llr(llr_ch, frozen_bits)
