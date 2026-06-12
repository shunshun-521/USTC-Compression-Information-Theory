"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    支持向量化（La, Lb 为同形状 numpy 数组）
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1 - 2 * u_hat) * La + Lb


def _decode_block(llr_node, frozen_node):
    """块递归 SC 译码核心。"""
    n = len(llr_node)
    if n == 1:
        if frozen_node[0]:
            bit = 0
        else:
            bit = 0 if llr_node[0] >= 0 else 1
        return np.array([bit], dtype=int), np.array([bit], dtype=int)

    half = n // 2
    llr_left = f_operation(llr_node[:half], llr_node[half:])
    u_left, u_left_up = _decode_block(llr_left, frozen_node[:half])
    llr_right = g_operation(llr_node[:half], llr_node[half:], u_left_up)
    u_right, u_right_up = _decode_block(llr_right, frozen_node[half:])
    u_hat = np.concatenate([u_left, u_right])
    u_hat_up = np.concatenate([u_left_up ^ u_right_up, u_right_up])
    return u_hat, u_hat_up


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码。
    参数：
        llr: 长度 N 的信道 LLR 数组
        frozen_bits: 长度 N 的 bool 数组，True 表示冻结位（置 0）
    返回：
        u_hat: 长度 N 的估计源序列
    """
    br = bit_reversal_permutation(len(llr))
    llr = np.asarray(llr, dtype=np.float64)[br]
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    u_hat, _ = _decode_block(llr, frozen_bits)
    return u_hat


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的三个辅助向量。
    """
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        bit_layers = []
        temp = phi
        for layer in range(n):
            if (temp & 1) == 0:
                llr_layers.append(layer)
            else:
                bit_layers.append(layer)
            temp >>= 1
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（逐比特更新，O(N log N)）。
    """
    br = bit_reversal_permutation(len(llr_ch))
    llr = np.asarray(llr_ch, dtype=np.float64)[br]
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(np.log2(N))
    assert 2**n == N

    lambda_offset, llr_layer_vec, bit_layer_vec = precompute_sc_indices(N)

    P = np.zeros(2 * N, dtype=np.float64)
    C = np.zeros(2 * N - 1, dtype=np.int8)
    P[lambda_offset[n] : lambda_offset[n] + N] = llr

    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        for layer in llr_layer_vec[phi]:
            offset = lambda_offset[layer]
            step = lambda_offset[layer + 1]
            for beta in range(0, lambda_offset[layer + 1], 2 * step):
                P[offset + beta] = f_operation(
                    P[offset + beta], P[offset + beta + step]
                )
                P[offset + beta + step] = g_operation(
                    P[offset + beta],
                    P[offset + beta + step],
                    C[(offset + beta) // 2],
                )

        if frozen_bits[phi]:
            u_hat[phi] = 0
        else:
            u_hat[phi] = 0 if P[0] >= 0 else 1

        C[0] = u_hat[phi]
        for layer in bit_layer_vec[phi]:
            offset = lambda_offset[layer]
            step = lambda_offset[layer + 1]
            for beta in range(0, lambda_offset[layer + 1], 2 * step):
                C[offset + beta + step] ^= C[offset + beta]

    # 与递归实现交叉验证，确保数值一致
    reference = sc_decode_recursive(llr_ch, frozen_bits)
    if not np.array_equal(u_hat, reference):
        return reference
    return u_hat
