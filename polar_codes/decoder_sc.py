"""
极化码 SC（串行抵消）译码器
基于 lazy LLR 计算（与标准 Arikan 因子图一致）
"""
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum f"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算，u_hat 为 0/1"""
    return La * (1 - 2 * u_hat) + Lb


def _B_check(ll, ii):
    return (ii // (1 << ll)) % 2


def _s_updater(ll, ii, s):
    if _B_check(ll - 1, ii):
        s[ll, ii] = s[ll - 1, ii]
    else:
        if s[ll - 1, ii] == -1:
            _s_updater(ll - 1, ii, s)
        if s[ll - 1, ii + (1 << (ll - 1))] == -1:
            _s_updater(ll - 1, ii + (1 << (ll - 1)), s)
        s[ll, ii] = s[ll - 1, ii] ^ s[ll - 1, ii + (1 << (ll - 1))]


def _Li(ll, ii, llrs, s, n):
    if llrs[ll, ii] != -np.inf:
        return llrs[ll, ii]
    if _B_check(ll, ii) == 0:
        llrs[ll, ii] = f_operation(
            _Li(ll + 1, ii, llrs, s, n),
            _Li(ll + 1, ii + (1 << ll), llrs, s, n),
        )
    else:
        if ll > 0:
            _s_updater(ll, ii - (1 << ll), s)
        llrs[ll, ii] = g_operation(
            _Li(ll + 1, ii - (1 << ll), llrs, s, n),
            _Li(ll + 1, ii, llrs, s, n),
            s[ll, ii - (1 << ll)],
        )
    return llrs[ll, ii]


def sc_decode(llr_ch, frozen_bits):
    """
    SC 译码。frozen_bits 中 1/True 表示冻结位。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    # 编码输出含 B_N；信道 LLR 需倒序对齐因子图
    llr_ch = llr_ch[br]
    frozen_bits = np.asarray(frozen_bits)
    n = int(np.log2(N))
    llrs = -np.inf * np.ones((n + 1, N), dtype=np.float64)
    llrs[n, :] = llr_ch
    s = -np.ones((n + 1, N), dtype=np.int8)
    info_mask = np.ones(N, dtype=np.int8)
    if frozen_bits.dtype == bool:
        info_mask[frozen_bits] = 0
    else:
        info_mask[frozen_bits.astype(int) > 0] = 0

    u_hat = np.zeros(N, dtype=int)
    for ii in range(N):
        if info_mask[ii] == 0:
            s[0, ii] = 0
            llrs[0, ii] = np.inf
            u_hat[ii] = 0
        else:
            llrs[0, ii] = _Li(0, ii, llrs, s, n)
            u_hat[ii] = 1 if llrs[0, ii] < 0 else 0
            s[0, ii] = u_hat[ii]
    return u_hat


def sc_decode_recursive(llr_ch, frozen_bits):
    """与 sc_decode 相同（接口兼容）"""
    return sc_decode(llr_ch, frozen_bits)


def precompute_sc_indices(N):
    n = int(np.log2(N))
    return list(range(n + 1)), [[] for _ in range(N)], [[] for _ in range(N)]
