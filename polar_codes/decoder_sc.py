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
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    u_hat = np.asarray(u_hat)
    return (1.0 - 2.0 * u_hat) * La + Lb


def _b_check(layer, idx):
    """判断因子图节点类型：0=f，1=g"""
    return (idx // (1 << layer)) % 2


def _s_updater(layer, idx, bits):
    """递归更新部分比特和"""
    if _b_check(layer - 1, idx):
        bits[layer, idx] = bits[layer - 1, idx]
    else:
        if bits[layer - 1, idx] == -1:
            _s_updater(layer - 1, idx, bits)
        sibling = idx + (1 << (layer - 1))
        if bits[layer - 1, sibling] == -1:
            _s_updater(layer - 1, sibling, bits)
        bits[layer, idx] = bits[layer - 1, idx] ^ bits[layer - 1, sibling]


def _compute_llr(layer, idx, llrs, bits):
    """惰性计算指定节点的 LLR"""
    if llrs[layer, idx] != -np.inf:
        return llrs[layer, idx]

    if _b_check(layer, idx) == 0:
        llrs[layer, idx] = f_operation(
            _compute_llr(layer + 1, idx, llrs, bits),
            _compute_llr(layer + 1, idx + (1 << layer), llrs, bits),
        )
    else:
        if layer > 0:
            _s_updater(layer, idx - (1 << layer), bits)
        left_idx = idx - (1 << layer)
        llrs[layer, idx] = g_operation(
            _compute_llr(layer + 1, left_idx, llrs, bits),
            _compute_llr(layer + 1, idx, llrs, bits),
            bits[layer, left_idx],
        )
    return llrs[layer, idx]


def _to_info_mask(frozen_bits):
    frozen_bits = np.asarray(frozen_bits)
    if frozen_bits.dtype == bool:
        return (~frozen_bits).astype(np.int8)
    return (1 - frozen_bits).astype(np.int8)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_block(llr_node, offset):
        n = len(llr_node)
        if n == 1:
            if frozen_bits[offset]:
                u_hat[offset] = 0
            else:
                u_hat[offset] = 0 if llr_node[0] >= 0 else 1
            return

        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        decode_block(llr_left, offset)
        u_left = u_hat[offset:offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        decode_block(llr_right, offset + half)

    decode_block(llr, 0)
    return u_hat


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
  """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        llr_layers = []
        l = 0
        while l < n and ((phi >> l) & 1):
            llr_layers.append(l)
            l += 1
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        l = 0
        while l < n:
            if ((phi >> l) & 1) == 0:
                bit_layers.append(l)
            l += 1
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（惰性 LLR 计算）。
    """
    from encoder import bit_reversal_permutation

    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(math.log2(N))
    info_mask = _to_info_mask(frozen_bits)

    br = bit_reversal_permutation(N)
    llr = llr_ch[br].copy()

    llrs = -np.inf * np.ones((n + 1, N), dtype=np.float64)
    llrs[-1, :] = llr
    bits = -np.ones((n + 1, N), dtype=np.int8)
    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        if info_mask[phi] == 0:
            u_hat[phi] = 0
            llrs[0, phi] = np.inf
            bits[0, phi] = 0
        else:
            llr_val = _compute_llr(0, phi, llrs, bits)
            u_hat[phi] = 1 if llr_val < 0 else 0
            bits[0, phi] = u_hat[phi]

    return u_hat


def sc_decode_layered(llr_ch, frozen_bits):
    """分层 SC 译码别名，与 sc_decode 相同。"""
    return sc_decode(llr_ch, frozen_bits)
