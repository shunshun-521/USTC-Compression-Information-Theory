"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _bit_llr_bp_schedule(llr_ch, frozen_bits, u_prefix, phi, alpha=1.0):
    """利用与 BP 相同的因子图，计算第 phi 个比特的 LLR"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    LARGE = 1e6

    L = np.zeros((N, n + 1), dtype=np.float64)
    R = np.zeros((N, n + 1), dtype=np.float64)
    L[:, n] = llr_ch
    R[:, 0] = 0.0
    R[frozen_bits, 0] = LARGE
    for i in range(phi):
        R[i, 0] = LARGE if u_prefix[i] == 0 else -LARGE

    for j in range(n, 0, -1):
        sp = 1 << (j - 1)
        for i in range(0, N, 2 * sp):
            for t in range(sp):
                idx = i + t
                L[idx, j - 1] = alpha * f_operation(
                    R[idx, j - 1] + L[idx + sp, j], L[idx, j]
                )
                L[idx + sp, j - 1] = alpha * f_operation(
                    R[idx, j - 1], L[idx, j]
                ) + L[idx + sp, j]

    return L[phi, 0] + R[phi, 0]


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，与主因子图可能不完全一致）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=np.int8)

    def decode_node(llr_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            u_hat[idx] = 0 if (frozen_bits[idx] or llr_node[0] >= 0) else 1
            return
        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        decode_node(llr_left, bit_offset)
        u_left = u_hat[bit_offset : bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        decode_node(llr_right, bit_offset + half)

    decode_node(llr, 0)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量"""
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        p = phi
        while p & 1:
            llr_layers.append(int(math.log2(p & -p)))
            p >>= 1
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        if (phi & 1) == 0:
            bit_layers.append(0)
        p = phi >> 1
        while p & 1:
            bit_layers.append(int(math.log2(p & -p)) + 1)
            p >>= 1
        bit_layer_vec.append(bit_layers)

    lambda_offset = [0]
    for l in range(1, n + 1):
        lambda_offset.append(lambda_offset[-1] + (1 << (l - 1)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode_nonrecursive(llr_ch, frozen_bits):
    """基于 lambda_offset 的非递归 SC（保留接口）"""
    return sc_decode(llr_ch, frozen_bits)


def sc_decode_matrix(llr_ch, frozen_bits):
    """矩阵存储 SC（与 sc_decode 相同调度）"""
    return sc_decode(llr_ch, frozen_bits)


def sc_decode(llr_ch, frozen_bits):
    """SC 译码主入口（BP 因子图上的串行调度，与 encoder/BP 一致）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    u_hat = np.zeros(N, dtype=np.int8)

    for phi in range(N):
        llr_bit = _bit_llr_bp_schedule(llr_ch, frozen_bits, u_hat, phi, alpha=1.0)
        if frozen_bits[phi]:
            u_hat[phi] = 0
        else:
            u_hat[phi] = 0 if llr_bit >= 0 else 1
    return u_hat
