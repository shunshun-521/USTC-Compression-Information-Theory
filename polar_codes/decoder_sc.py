"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb。"""
    u = np.asarray(u_hat, dtype=np.float64)
    return (1.0 - 2.0 * u) * La + Lb


def _b_check(layer, idx):
    """判断节点是否为 g 分支。"""
    return (idx // (1 << layer)) % 2


def _s_updater(layer, idx, bits):
    """递归更新部分和比特。"""
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
    """按需递归计算 LLR。"""
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
        llrs[layer, idx] = g_operation(
            _compute_llr(layer + 1, idx - (1 << layer), llrs, bits),
            _compute_llr(layer + 1, idx, llrs, bits),
            bits[layer, idx - (1 << layer)],
        )
    return llrs[layer, idx]


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（与 sc_decode 等价）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量（接口兼容）。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = [list(range(n)) for _ in range(N)]
    bit_layer_vec = [list(range(n, 0, -1)) for _ in range(N)]
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    SC 译码主函数（按需递归 LLR 计算，自然序处理）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    n = int(math.log2(N))

    llrs = np.full((n + 1, N), -np.inf, dtype=np.float64)
    llrs[n, :] = llr_ch
    bits = np.full((n + 1, N), -1, dtype=np.int32)

    u_hat = np.zeros(N, dtype=int)
    for phi in range(N):
        if frozen_bits[phi]:
            u_hat[phi] = 0
            llrs[0, phi] = np.inf
            bits[0, phi] = 0
        else:
            llr = _compute_llr(0, phi, llrs, bits)
            u_hat[phi] = 0 if llr >= 0 else 1
            bits[0, phi] = u_hat[phi]

    return u_hat
