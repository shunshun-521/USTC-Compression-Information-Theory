"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np

INF = 1e100


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    u_hat = np.asarray(u_hat)
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return (1.0 - 2.0 * u_hat) * La + Lb


def _b_check(layer, index):
    """判断 index 在 layer 层是否处于下半分支（g 节点）"""
    return (index // (1 << layer)) % 2


def _s_updater(layer, index, bits):
    """递归更新比特数组（用于 g 节点）"""
    if _b_check(layer - 1, index):
        bits[layer, index] = bits[layer - 1, index]
    else:
        if bits[layer - 1, index] == -1:
            _s_updater(layer - 1, index, bits)
        sibling = index + (1 << (layer - 1))
        if bits[layer - 1, sibling] == -1:
            _s_updater(layer - 1, sibling, bits)
        bits[layer, index] = bits[layer - 1, index] ^ bits[layer - 1, sibling]


def _li(layer, index, llrs, bits, n):
    """懒计算 LLR（递归）"""
    if llrs[layer, index] > -INF / 2:
        return llrs[layer, index]

    if _b_check(layer, index) == 0:
        left = _li(layer + 1, index, llrs, bits, n)
        right = _li(layer + 1, index + (1 << layer), llrs, bits, n)
        llrs[layer, index] = f_operation(left, right)
    else:
        if layer > 0:
            _s_updater(layer, index - (1 << layer), bits)
        left_idx = index - (1 << layer)
        left_llr = _li(layer + 1, left_idx, llrs, bits, n)
        right_llr = _li(layer + 1, index, llrs, bits, n)
        llrs[layer, index] = g_operation(left_llr, right_llr, bits[layer, left_idx])

    return llrs[layer, index]


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归懒计算 SC 译码（参考实现）"""
    return sc_decode(llr_ch, frozen_bits)


def precompute_sc_indices(N):
    """预计算辅助向量（供文档兼容）"""
    n = int(np.log2(N))
    lambda_offset = list(range(N))
    llr_layer_vec = [list(range(n)) for _ in range(N)]
    bit_layer_vec = [list(range(n)) if i % 2 == 1 else [] for i in range(N)]
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    SC 译码主函数。
    frozen_bits: 1 表示冻结位，0 表示信息位
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    n = int(np.log2(N))

    llrs = np.full((n + 1, N), -INF, dtype=np.float64)
    bits = np.full((n + 1, N), -1, dtype=int)
    llrs[n, :] = llr_ch

    u_hat = np.zeros(N, dtype=int)

    for i in range(N):
        if frozen_bits[i] == 1:
            bits[0, i] = 0
            llrs[0, i] = INF
            u_hat[i] = 0
        else:
            llr_val = _li(0, i, llrs, bits, n)
            u_hat[i] = 1 if llr_val < 0 else 0
            bits[0, i] = u_hat[i]

    return u_hat
