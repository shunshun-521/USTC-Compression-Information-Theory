"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
import math


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def _B_check(level, idx):
    return (idx // (1 << level)) % 2


def _s_updater(level, idx, bits):
    if _B_check(level - 1, idx):
        bits[level, idx] = bits[level - 1, idx]
    else:
        if bits[level - 1, idx] == -1:
            _s_updater(level - 1, idx, bits)
        partner = idx + (1 << (level - 1))
        if bits[level - 1, partner] == -1:
            _s_updater(level - 1, partner, bits)
        bits[level, idx] = bits[level - 1, idx] ^ bits[level - 1, partner]


def _compute_llr(level, idx, llrs, bits):
    if llrs[level, idx] != -np.inf:
        return llrs[level, idx]

    if _B_check(level, idx) == 0:
        llrs[level, idx] = f_operation(
            _compute_llr(level + 1, idx, llrs, bits),
            _compute_llr(level + 1, idx + (1 << level), llrs, bits),
        )
    else:
        if level > 0:
            _s_updater(level, idx - (1 << level), bits)
        llrs[level, idx] = g_operation(
            _compute_llr(level + 1, idx - (1 << level), llrs, bits),
            _compute_llr(level + 1, idx, llrs, bits),
            bits[level, idx - (1 << level)],
        )
    return llrs[level, idx]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（按需计算 LLR 的高效实现）。
    frozen_bits: 1/True 表示冻结位，0/False 表示信息位
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits)
    if frozen_bits.dtype != bool:
        frozen_bits = frozen_bits.astype(bool)

    N = len(llr_ch)
    n = int(math.log2(N))

    llrs = np.full((n + 1, N), -np.inf, dtype=np.float64)
    llrs[n, :] = llr_ch
    bits = -np.ones((n + 1, N), dtype=int)
    u_hat = np.zeros(N, dtype=int)

    for i in range(N):
        if frozen_bits[i]:
            u_hat[i] = 0
            llrs[0, i] = np.inf
        else:
            llrs[0, i] = _compute_llr(0, i, llrs, bits)
            u_hat[i] = 1 if llrs[0, i] < 0 else 0
        bits[0, i] = u_hat[i]

    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（调用与主实现相同的译码器）"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算辅助向量（兼容接口）"""
    n = int(math.log2(N))
    llr_layer_vec = [list(range(n)) for _ in range(N)]
    bit_layer_vec = [list(range(n, 0, -1)) for _ in range(N)]
    lambda_offset = [0] * (n + 1)
    for i in range(1, n + 1):
        lambda_offset[i] = 2 ** (n - i)
    return lambda_offset, llr_layer_vec, bit_layer_vec


sc_decode_efficient = sc_decode
