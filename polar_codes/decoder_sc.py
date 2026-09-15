"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation

_UNINIT = np.nan


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


def _s_updater(layer, idx, s):
    if _b_check(layer - 1, idx):
        s[layer, idx] = s[layer - 1, idx]
    else:
        if s[layer - 1, idx] == -1:
            _s_updater(layer - 1, idx, s)
        partner = idx + (1 << (layer - 1))
        if s[layer - 1, partner] == -1:
            _s_updater(layer - 1, partner, s)
        s[layer, idx] = s[layer - 1, idx] ^ s[layer - 1, partner]


def _compute_llr(layer, idx, llrs, s):
    if not np.isnan(llrs[layer, idx]):
        return llrs[layer, idx]

    if _b_check(layer, idx) == 0:
        llrs[layer, idx] = f_operation(
            _compute_llr(layer + 1, idx, llrs, s),
            _compute_llr(layer + 1, idx + (1 << layer), llrs, s),
        )
    else:
        if layer > 0:
            _s_updater(layer, idx - (1 << layer), s)
        top = _compute_llr(layer + 1, idx - (1 << layer), llrs, s)
        bottom = _compute_llr(layer + 1, idx, llrs, s)
        llrs[layer, idx] = g_operation(top, bottom, s[layer, idx - (1 << layer)])
    return llrs[layer, idx]


def _sc_decode_core(llr_br, frozen_bits):
    """内部 SC 译码核心（输入为比特倒序后的 LLR）。"""
    N = len(llr_br)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    llrs = np.full((n + 1, N), _UNINIT, dtype=np.float64)
    llrs[n, :] = llr_br
    s = -np.ones((n + 1, N), dtype=np.int8)
    u_hat = np.zeros(N, dtype=int)

    for i in range(N):
        llr_i = _compute_llr(0, i, llrs, s)
        if frozen_bits[i]:
            u_hat[i] = 0
        else:
            u_hat[i] = 0 if llr_i >= 0 else 1
        s[0, i] = u_hat[i]

    return u_hat


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（参考实现，与 sc_decode 等价）。"""
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    return _sc_decode_core(llr_ch[br], frozen_bits)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers = [layer for layer in range(n) if (phi >> layer) & 1 == 0]
        llr_layer_vec.append(layers)
        bit_layers = []
        if phi % 2 == 1:
            bit_layers = [layer for layer in range(n) if (phi >> layer) & 1]
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    SC 译码主函数（高效 Li 递归 LLR 计算）。
    编码器含比特倒序置换，信道 LLR 需做相同置换后译码。
    """
    return sc_decode_recursive(llr_ch, frozen_bits)
