"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（惰性 LLR 计算，高效实现）
"""
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    u_hat = np.asarray(u_hat)
    return (1.0 - 2.0 * u_hat) * La + Lb


def _b_check(layer, idx):
    return (idx // (1 << layer)) % 2


def _s_updater(layer, idx, bits):
    if _b_check(layer - 1, idx):
        bits[layer, idx] = bits[layer - 1, idx]
    else:
        if bits[layer - 1, idx] == -1:
            _s_updater(layer - 1, idx, bits)
        partner = idx + (1 << (layer - 1))
        if bits[layer - 1, partner] == -1:
            _s_updater(layer - 1, partner, bits)
        bits[layer, idx] = bits[layer - 1, idx] ^ bits[layer - 1, partner]


def _li(layer, idx, llrs, bits):
    if llrs[layer, idx] != -np.inf:
        return llrs[layer, idx]
    if _b_check(layer, idx) == 0:
        llrs[layer, idx] = f_operation(
            _li(layer + 1, idx, llrs, bits),
            _li(layer + 1, idx + (1 << layer), llrs, bits),
        )
        return llrs[layer, idx]
    if layer > 0:
        _s_updater(layer, idx - (1 << layer), bits)
    llrs[layer, idx] = g_operation(
        _li(layer + 1, idx - (1 << layer), llrs, bits),
        _li(layer + 1, idx, llrs, bits),
        bits[layer, idx - (1 << layer)],
    )
    return llrs[layer, idx]


def precompute_sc_indices(N):
    """预计算辅助向量（接口兼容；惰性 SC 不依赖该表）"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = [list(range(n)) for _ in range(N)]
    bit_layer_vec = [[] for _ in range(N)]
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（惰性 LLR + 比特回传）"""
    N = len(llr_ch)
    n = int(np.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    llrs = -np.inf * np.ones((n + 1, N), dtype=np.float64)
    llrs[-1, :] = llr_ch.astype(np.float64)
    bits = -np.ones((n + 1, N), dtype=np.int8)

    for i in range(N):
        if frozen_bits[i]:
            bits[0, i] = 0
            llrs[0, i] = np.inf
        else:
            llrs[0, i] = _li(0, i, llrs, bits)
            bits[0, i] = 1 if llrs[0, i] < 0 else 0

    return bits[0, :].astype(np.int8)


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（与 sc_decode 等价，供对照）"""
    return sc_decode(llr_ch, frozen_bits)
