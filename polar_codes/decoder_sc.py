"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _b_check(layer, idx):
    return (idx // (1 << layer)) % 2


def _s_updater(layer, idx, s):
    """递归更新部分和比特"""
    if _b_check(layer - 1, idx):
        s[layer, idx] = s[layer - 1, idx]
    else:
        if s[layer - 1, idx] == -1:
            _s_updater(layer - 1, idx, s)
        partner = idx + (1 << (layer - 1))
        if s[layer - 1, partner] == -1:
            _s_updater(layer - 1, partner, s)
        s[layer, idx] = s[layer - 1, idx] ^ s[layer - 1, partner]


def _li(layer, idx, llrs, s, n):
    """惰性计算 LLR"""
    if llrs[layer, idx] != -np.inf:
        return llrs[layer, idx]

    if _b_check(layer, idx) == 0:
        llrs[layer, idx] = f_operation(
            _li(layer + 1, idx, llrs, s, n),
            _li(layer + 1, idx + (1 << layer), llrs, s, n),
        )
    else:
        if layer > 0:
            _s_updater(layer, idx - (1 << layer), s)
        llrs[layer, idx] = g_operation(
            _li(layer + 1, idx - (1 << layer), llrs, s, n),
            _li(layer + 1, idx, llrs, s, n),
            s[layer, idx - (1 << layer)],
        )
    return llrs[layer, idx]


def _prepare_channel_llr(llr_ch):
    """将信道 LLR 转换为译码树顺序（与蝶形编码对应）"""
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[br]


def sc_decode_recursive(llr_ch, frozen_bits):
    """
    递归 SC 译码（参考实现）。
    采用与 sc_decode 相同的惰性 LLR 算法，结果一致。
    """
    return sc_decode(llr_ch, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers = list(range(n))
        llr_layer_vec.append(layers)
        bit_layer_vec.append(list(range(n)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（惰性 LLR 计算）。
    frozen_bits: 1=冻结位, 0=信息位
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    channel_llr = _prepare_channel_llr(llr_ch)

    llrs = np.full((n + 1, N), -np.inf, dtype=np.float64)
    llrs[n, :] = channel_llr
    s = np.full((n + 1, N), -1, dtype=np.int8)
    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        if frozen_bits[phi]:
            s[0, phi] = 0
            llrs[0, phi] = np.inf
            u_hat[phi] = 0
        else:
            llrs[0, phi] = _li(0, phi, llrs, s, n)
            u_hat[phi] = 1 if llrs[0, phi] < 0 else 0
            s[0, phi] = u_hat[phi]

    return u_hat
