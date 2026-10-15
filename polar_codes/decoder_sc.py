"""
极化码 SC（串行抵消）译码器
提供递归 LLR 计算（参考）与非递归逐比特 SC（主实现）
"""
import numpy as np


def f_operation(La, Lb):
    """min-sum f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    return (1 - 2 * u_hat) * La + Lb


def _f_node_minsum(a, b):
    return np.sign(a * b) * min(abs(a), abs(b))


def _g_node(llr1, llr2, s):
    return llr1 * (1 - 2 * s) + llr2


def _b_check(level, idx):
    return (idx // (1 << level)) % 2


def _s_updater(level, idx, s):
    if _b_check(level - 1, idx):
        s[level, idx] = s[level - 1, idx]
    else:
        if s[level - 1, idx] == -1:
            _s_updater(level - 1, idx, s)
        if s[level - 1, idx + (1 << (level - 1))] == -1:
            _s_updater(level - 1, idx + (1 << (level - 1)), s)
        s[level, idx] = s[level - 1, idx] ^ s[level - 1, idx + (1 << (level - 1))]


def _li(level, idx, llrs, s):
    if llrs[level, idx] != -np.inf:
        return llrs[level, idx]
    if _b_check(level, idx) == 0:
        llrs[level, idx] = _f_node_minsum(
            _li(level + 1, idx, llrs, s),
            _li(level + 1, idx + (1 << level), llrs, s),
        )
        return llrs[level, idx]
    if level > 0:
        _s_updater(level, idx - (1 << level), s)
    llrs[level, idx] = _g_node(
        _li(level + 1, idx - (1 << level), llrs, s),
        _li(level + 1, idx, llrs, s),
        s[level, idx - (1 << level)],
    )
    return llrs[level, idx]


def precompute_sc_indices(N):
    n = int(np.log2(N))
    lambda_offset = [1 << (n - i) for i in range(n + 1)]
    return lambda_offset, [[] for _ in range(N)], [[] for _ in range(N)]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归串行抵消译码（on-demand LLR，与 encoder 的 F^{\otimes n} 配套）。
    frozen_bits: True/1 表示冻结位。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits)
    if frozen_bits.dtype != bool:
        frozen_bits = frozen_bits.astype(bool)

    N = len(llr_ch)
    n = int(np.log2(N))
    llrs = np.full((n + 1, N), -np.inf, dtype=np.float64)
    llrs[n, :] = llr_ch
    s = np.full((n + 1, N), -1, dtype=np.int8)

    u_hat = np.zeros(N, dtype=int)
    for bit_idx in range(N):
        if frozen_bits[bit_idx]:
            s[0, bit_idx] = 0
            llrs[0, bit_idx] = np.inf
            u_hat[bit_idx] = 0
        else:
            llr0 = _li(0, bit_idx, llrs, s)
            u_hat[bit_idx] = 1 if llr0 < 0 else 0
            s[0, bit_idx] = u_hat[bit_idx]
    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """与 sc_decode 相同（保留接口）。"""
    return sc_decode(llr, frozen_bits)


def sc_path_metric_penalty(llr, u_bit):
    u_from_llr = 1 if llr < 0 else 0
    return 0.0 if u_bit == u_from_llr else abs(llr)
