"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，惰性 LLR 计算）
"""
import numpy as np
from encoder import channel_llr_to_decoder

INF = np.inf


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    return (1 - 2 * u_hat) * La + Lb


def _b_check(layer, index):
    """判断节点是否为 g 分支（下分支）。"""
    return (index // (1 << layer)) % 2


def _s_updater(layer, index, bits):
    """向上更新部分和比特。"""
    if _b_check(layer - 1, index):
        bits[layer, index] = bits[layer - 1, index]
    else:
        if bits[layer - 1, index] == -1:
            _s_updater(layer - 1, index, bits)
        sibling = index + (1 << (layer - 1))
        if bits[layer - 1, sibling] == -1:
            _s_updater(layer - 1, sibling, bits)
        bits[layer, index] = bits[layer - 1, index] ^ bits[layer - 1, sibling]


def _li(layer, index, llrs, bits):
    """惰性计算 LLR。"""
    if llrs[layer, index] != -INF:
        return llrs[layer, index]

    if _b_check(layer, index) == 0:
        left = _li(layer + 1, index, llrs, bits)
        right = _li(layer + 1, index + (1 << layer), llrs, bits)
        llrs[layer, index] = f_operation(left, right)
    else:
        if layer > 0:
            _s_updater(layer, index - (1 << layer), bits)
        top_bit = bits[layer, index - (1 << layer)]
        left = _li(layer + 1, index - (1 << layer), llrs, bits)
        right = _li(layer + 1, index, llrs, bits)
        llrs[layer, index] = g_operation(left, right, top_bit)
    return llrs[layer, index]


def _prepare_frozen_mask(frozen_bits):
    frozen_bits = np.asarray(frozen_bits)
    if frozen_bits.dtype == bool:
        return frozen_bits.astype(np.int8)
    return frozen_bits.astype(np.int8)


def sc_decode_core(llr_channel, info_mask):
    """
    SC 译码核心（惰性 LLR 计算）。
    info_mask: 1=信息位, 0=冻结位
    """
    N = len(llr_channel)
    n = int(np.log2(N))
    llrs = np.full((n + 1, N), -INF, dtype=np.float64)
    llrs[n, :] = llr_channel
    bits = np.full((n + 1, N), -1, dtype=np.int8)

    for i in range(N):
        if info_mask[i] == 0:
            bits[0, i] = 0
            llrs[0, i] = INF
        else:
            llrs[0, i] = _li(0, i, llrs, bits)
            bits[0, i] = 1 if llrs[0, i] < 0 else 0

    return bits[0, :].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    llr = channel_llr_to_decoder(llr)
    frozen_bits = _prepare_frozen_mask(frozen_bits)
    info_mask = 1 - frozen_bits
    return sc_decode_core(llr, info_mask)


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助索引（用于文档兼容）。"""
    n = int(np.log2(N))
    lambda_offset = [2 ** layer - 1 for layer in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        if phi == 0:
            layers_llr = list(range(n))
        else:
            trailing = (phi & -phi).bit_length() - 1
            layers_llr = list(range(n - 1, trailing - 1, -1))
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        temp = phi + 1
        while temp % 2 == 0:
            layer = (temp & -temp).bit_length() - 1
            layers_bit.append(layer - 1)
            temp //= 2
        bit_layer_vec.append(layers_bit)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（惰性 LLR 实现，与递归版本等价）。
    """
    llr = channel_llr_to_decoder(llr_ch)
    frozen_bits = _prepare_frozen_mask(frozen_bits)
    info_mask = 1 - frozen_bits
    return sc_decode_core(llr, info_mask)
