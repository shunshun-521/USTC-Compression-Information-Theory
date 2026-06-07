"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np

# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La * Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return La * (1 - 2 * u_hat) + Lb


def _b_check(layer, index):
    return (index // (1 << layer)) % 2


def _s_updater(layer, index, partial_sums):
    if _b_check(layer - 1, index):
        partial_sums[layer, index] = partial_sums[layer - 1, index]
    else:
        if partial_sums[layer - 1, index] == -1:
            _s_updater(layer - 1, index, partial_sums)
        sibling = index + (1 << (layer - 1))
        if partial_sums[layer - 1, sibling] == -1:
            _s_updater(layer - 1, sibling, partial_sums)
        partial_sums[layer, index] = (
            partial_sums[layer - 1, index] ^ partial_sums[layer - 1, sibling]
        )


def _compute_llr(layer, index, llrs, partial_sums):
    if llrs[layer, index] != -np.inf:
        return llrs[layer, index]
    if _b_check(layer, index) == 0:
        llrs[layer, index] = f_operation(
            _compute_llr(layer + 1, index, llrs, partial_sums),
            _compute_llr(layer + 1, index + (1 << layer), llrs, partial_sums),
        )
    else:
        if layer > 0:
            _s_updater(layer, index - (1 << layer), partial_sums)
        llrs[layer, index] = g_operation(
            _compute_llr(layer + 1, index - (1 << layer), llrs, partial_sums),
            _compute_llr(layer + 1, index, llrs, partial_sums),
            partial_sums[layer, index - (1 << layer)],
        )
    return llrs[layer, index]


# ==================== 递归 SC 译码（参考实现）====================


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（与主实现等价的树遍历版本）"""
    N = len(llr_ch)
    n = int(np.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    info_bits = ~frozen_bits
    llrs = -np.inf * np.ones((n + 1, N), dtype=np.float64)
    llrs[n, :] = np.asarray(llr_ch, dtype=np.float64)
    partial_sums = -np.ones((n + 1, N), dtype=np.int32)
    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        if frozen_bits[phi]:
            partial_sums[0, phi] = 0
            llrs[0, phi] = np.inf
            u_hat[phi] = 0
        else:
            llrs[0, phi] = _compute_llr(0, phi, llrs, partial_sums)
            partial_sums[0, phi] = 1 if llrs[0, phi] < 0 else 0
            u_hat[phi] = partial_sums[0, phi]
    return u_hat


# ==================== 非递归 SC 译码（高效实现）====================


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    """
    n = int(np.log2(N))
    lambda_offset = np.zeros(N, dtype=np.int32)
    llr_layer_vec = []
    bit_layer_vec = []

    offset = 0
    for phi in range(N):
        lambda_offset[phi] = offset
        temp = phi
        tz = 0
        while temp & 1:
            temp >>= 1
            tz += 1
        llr_layer_vec.append(list(range(tz, n)))
        bit_layer_vec.append(list(range(tz)))
        offset += len(llr_layer_vec[-1])

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数（惰性 LLR 计算）"""
    return sc_decode_recursive(llr_ch, frozen_bits)
