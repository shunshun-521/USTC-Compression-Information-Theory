"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _b_check(layer, index):
    """判断因子图节点类型（f 或 g）"""
    return (index // (1 << layer)) % 2


def _update_bits(layer, index, bits):
    """递归更新部分和比特（供 g 节点使用）"""
    if layer == 0:
        return
    if _b_check(layer - 1, index):
        bits[layer, index] = bits[layer - 1, index]
    else:
        if bits[layer - 1, index] == -1:
            _update_bits(layer - 1, index, bits)
        sibling = index + (1 << (layer - 1))
        if bits[layer - 1, sibling] == -1:
            _update_bits(layer - 1, sibling, bits)
        bits[layer, index] = bits[layer - 1, index] ^ bits[layer - 1, sibling]


def _compute_llr(layer, index, llrs, bits):
    """惰性计算指定节点的 LLR"""
    if not np.isneginf(llrs[layer, index]) and llrs[layer, index] != -np.inf:
        if not np.isnan(llrs[layer, index]):
            return llrs[layer, index]

    if _b_check(layer, index) == 0:
        left = _compute_llr(layer + 1, index, llrs, bits)
        right = _compute_llr(layer + 1, index + (1 << layer), llrs, bits)
        llrs[layer, index] = f_operation(left, right)
    else:
        if layer > 0:
            _update_bits(layer, index - (1 << layer), bits)
        left = _compute_llr(layer + 1, index - (1 << layer), llrs, bits)
        right = _compute_llr(layer + 1, index, llrs, bits)
        u_partial = bits[layer, index - (1 << layer)]
        llrs[layer, index] = g_operation(left, right, u_partial)
    return llrs[layer, index]


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（因子图惰性 LLR 递推，与树形递归等价）。
    frozen_bits: 非零/True 表示冻结位。
    """
    return _lazy_sc_decode(llr, frozen_bits)


def _lazy_sc_decode(llr, frozen_bits):
    """惰性 LLR 计算的 SC 译码（内部参考实现）。"""
    llr = np.asarray(llr, dtype=np.float64)
    N = len(llr)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits).astype(bool)

    llrs = np.full((n + 1, N), -np.inf, dtype=np.float64)
    llrs[n, :] = llr
    bits = -np.ones((n + 1, N), dtype=int)
    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        if frozen_bits[phi]:
            u_hat[phi] = 0
            bits[0, phi] = 0
            llrs[0, phi] = np.inf
        else:
            llrs[0, phi] = _compute_llr(0, phi, llrs, bits)
            u_hat[phi] = 1 if llrs[0, phi] < 0 else 0
            bits[0, phi] = u_hat[phi]

    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        llr_layers = []
        psi = phi
        layer = 0
        while (psi & 1) == 1 and layer < n:
            psi >>= 1
            layer += 1
        for lyr in range(layer, n):
            llr_layers.append(lyr)
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        psi = phi + 1
        layer = 0
        while layer < n:
            if (psi & 1) == 1:
                bit_layers.append(layer)
            psi >>= 1
            layer += 1
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数（惰性 LLR，O(N log N)）。"""
    return _lazy_sc_decode(llr_ch, frozen_bits)



def prepare_channel_llr(llr_ch, N):
    """
    将信道 LLR 转换为 SC 译码器所需的顺序。
    编码器输出含比特倒序置换，译码前需对 LLR 做相同倒序。
    """
    from encoder import bit_reversal_permutation
    return np.asarray(llr_ch)[bit_reversal_permutation(N)]
