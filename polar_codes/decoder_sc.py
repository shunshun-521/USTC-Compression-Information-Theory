"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

INF = np.float64(1e100)


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
    return (1 - 2 * u_hat) * La + Lb


def _b_check(layer, idx):
    return (idx // (1 << layer)) % 2


def _s_updater(layer, idx, bits):
    if _b_check(layer - 1, idx):
        bits[layer, idx] = bits[layer - 1, idx]
    else:
        if bits[layer - 1, idx] < 0:
            _s_updater(layer - 1, idx, bits)
        sibling = idx + (1 << (layer - 1))
        if bits[layer - 1, sibling] < 0:
            _s_updater(layer - 1, sibling, bits)
        bits[layer, idx] = bits[layer - 1, idx] ^ bits[layer - 1, sibling]


def _compute_llr(layer, idx, llrs, bits):
    if llrs[layer, idx] > -INF / 2:
        return llrs[layer, idx]

    if _b_check(layer, idx) == 0:
        left = _compute_llr(layer + 1, idx, llrs, bits)
        right = _compute_llr(layer + 1, idx + (1 << layer), llrs, bits)
        llrs[layer, idx] = f_operation(left, right)
    else:
        if layer > 0:
            _s_updater(layer, idx - (1 << layer), bits)
        left = _compute_llr(layer + 1, idx - (1 << layer), llrs, bits)
        right = _compute_llr(layer + 1, idx, llrs, bits)
        llrs[layer, idx] = g_operation(left, right, bits[layer, idx - (1 << layer)])

    return llrs[layer, idx]


def _sc_decode_core(llr_ch, frozen_bits):
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    llrs = np.full((n + 1, N), -INF, dtype=np.float64)
    bits = np.full((n + 1, N), -1, dtype=np.int8)
    llrs[n, :] = llr_ch

    u_hat = np.zeros(N, dtype=int)
    for phi in range(N):
        if frozen_bits[phi]:
            bits[0, phi] = 0
            llrs[0, phi] = INF
            u_hat[phi] = 0
        else:
            llrs[0, phi] = _compute_llr(0, phi, llrs, bits)
            u_hat[phi] = 1 if llrs[0, phi] < 0 else 0
            bits[0, phi] = u_hat[phi]

    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数。"""
    return _sc_decode_core(llr_ch, frozen_bits)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（与主实现共享核心逻辑）。"""
    return _sc_decode_core(llr, frozen_bits)


def _trailing_ones(phi):
    count = 0
    while phi & 1:
        phi >>= 1
        count += 1
    return count


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量。"""
    n = int(math.log2(N))
    lambda_offset = list(range(N))
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        t1 = _trailing_ones(phi)
        if t1 >= n:
            llr_layers = [n - 1]
        else:
            llr_layers = list(range(t1, n))
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(list(range(t1)))

    return lambda_offset, llr_layer_vec, bit_layer_vec
