"""
极化码 SC（串行抵消）译码器
自然序惰性 SC 实现，与蝶形+比特倒序编码器配套
"""
import numpy as np

from encoder import bit_reversal_permutation


def _prepare_channel_llr(llr_ch):
    """将比特倒序码字 LLR 映射到蝶形因子图自然序"""
    N = len(llr_ch)
    return llr_ch[np.argsort(bit_reversal_permutation(N))]


# ==================== 基本运算 ====================

def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def _b_check(level, idx):
    return (idx // (1 << level)) % 2


def _s_updater(level, idx, s):
    if _b_check(level - 1, idx):
        s[level, idx] = s[level - 1, idx]
    else:
        if s[level - 1, idx] < 0:
            _s_updater(level - 1, idx, s)
        partner = idx + (1 << (level - 1))
        if s[level - 1, partner] < 0:
            _s_updater(level - 1, partner, s)
        s[level, idx] = s[level - 1, idx] ^ s[level - 1, partner]


def _compute_llr(level, idx, llrs, s):
    if llrs[level, idx] != -np.inf:
        return llrs[level, idx]
    if _b_check(level, idx) == 0:
        llrs[level, idx] = f_operation(
            _compute_llr(level + 1, idx, llrs, s),
            _compute_llr(level + 1, idx + (1 << level), llrs, s),
        )
    else:
        if level > 0:
            _s_updater(level, idx - (1 << level), s)
        partner = idx - (1 << level)
        prev_bit = s[level, partner] if level > 0 else s[0, partner]
        llrs[level, idx] = g_operation(
            _compute_llr(level + 1, partner, llrs, s),
            _compute_llr(level + 1, idx, llrs, s),
            prev_bit,
        )
    return llrs[level, idx]


# ==================== 递归 SC 译码（参考实现）====================

def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码"""
    return sc_decode(llr, frozen_bits)


# ==================== 非递归 SC 译码 ====================

def precompute_sc_indices(N):
    """预计算辅助向量（供 SCL 使用）"""
    n = int(np.log2(N))
    layers = list(range(n))
    return [1 << i for i in range(n + 1)], [layers[:] for _ in range(N)], [layers[::-1] for _ in range(N)]


def sc_decode(llr_ch, frozen_bits):
    """自然序 SC 译码主函数"""
    llr_ch = _prepare_channel_llr(llr_ch)
    N = len(llr_ch)
    n = int(np.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    llrs = np.full((n + 1, N), -np.inf, dtype=np.float64)
    llrs[n, :] = llr_ch
    s = np.full((n + 1, N), -1, dtype=np.int8)
    u_hat = np.zeros(N, dtype=int)

    for idx in range(N):
        llr_val = _compute_llr(0, idx, llrs, s)
        if frozen_bits[idx]:
            u_hat[idx] = 0
            s[0, idx] = 0
        else:
            u_hat[idx] = 0 if llr_val >= 0 else 1
            s[0, idx] = u_hat[idx]

    return u_hat
