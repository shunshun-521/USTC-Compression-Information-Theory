"""
SC / SCL 译码核心（惰性 LLR 计算，与标准极化蝶形编码配套）。
参考实现思路：OkanErturk16/Polar-Code polarLib.py
"""
import numpy as np

try:
    import numba

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


def _f_node_minsum(a, b):
    return np.sign(a * b) * np.minimum(np.abs(a), np.abs(b))


def _g_node(llr1, llr2, s):
    return llr1 * (1 - 2 * s) + llr2


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


def _Li(ll, ii, llrs, s):
    if llrs[ll, ii] != -np.inf:
        return llrs[ll, ii]
    if _B_check(ll, ii) == 0:
        llrs[ll, ii] = _f_node_minsum(
            _Li(ll + 1, ii, llrs, s), _Li(ll + 1, ii + (1 << ll), llrs, s)
        )
        return llrs[ll, ii]
    if ll > 0:
        _s_updater(ll, ii - (1 << ll), s)
    llrs[ll, ii] = _g_node(
        _Li(ll + 1, ii - (1 << ll), llrs, s),
        _Li(ll + 1, ii, llrs, s),
        s[ll, ii - (1 << ll)],
    )
    return llrs[ll, ii]


def sc_decoder_impl(llr_channel, if_information_bit):
    """llr_channel 已与编码器比特倒序对齐。"""
    llr_channel = np.asarray(llr_channel, dtype=np.float32)
    if_information_bit = np.asarray(if_information_bit, dtype=np.int8)
    N = len(llr_channel)
    n = int(np.log2(N))

    llrs = -np.inf * np.ones((n + 1, 1 << n), dtype=np.float32)
    llrs[-1, :] = llr_channel
    s = -1 * np.ones((n + 1, 1 << n), dtype=np.int8)

    for ii in range(N):
        if if_information_bit[ii] == 0:
            s[0, ii] = 0
            llrs[0, ii] = np.inf
        else:
            llrs[0, ii] = _Li(0, ii, llrs, s)
            s[0, ii] = 1 if llrs[0, ii] < 0 else 0
    return s[0, :].copy()


def scl_decoder_impl(llr_channel, if_information_bit, list_size):
    llr_channel = np.asarray(llr_channel, dtype=np.float32)
    if_information_bit = np.asarray(if_information_bit, dtype=np.int8)
    N = len(llr_channel)
    n = int(np.log2(N))
    L = int(list_size)

    llrs_list = []
    s_list = []
    for _ in range(L):
        llrs = -np.inf * np.ones((n + 1, 1 << n), dtype=np.float32)
        llrs[-1, :] = llr_channel
        llrs_list.append(llrs)
        s_list.append(-1 * np.ones((n + 1, 1 << n), dtype=np.int8))

    PM = np.inf * np.ones(L, dtype=np.float32)
    PM[0] = 0.0
    DM = np.zeros(L, dtype=np.float32)
    PM_DM = np.zeros(2 * L, dtype=np.float32)

    for ii in range(N):
        if if_information_bit[ii] == 0:
            for dd in range(L):
                llrs_list[dd][0, ii] = _Li(0, ii, llrs_list[dd], s_list[dd])
                s_list[dd][0, ii] = 0
                PM[dd] += -llrs_list[dd][0, ii] * (llrs_list[dd][0, ii] < 0)
        else:
            for dd in range(L):
                llrs_list[dd][0, ii] = _Li(0, ii, llrs_list[dd], s_list[dd])
                s_list[dd][0, ii] = 1 if llrs_list[dd][0, ii] < 0 else 0
                DM[dd] = np.abs(llrs_list[dd][0, ii])

            if L > 1:
                PM_DM[:L] = PM
                PM_DM[L:] = PM + DM
                idx_sort = np.argsort(PM_DM)
                idx_min_low = idx_sort[:L][idx_sort[:L] >= L] - L
                idx_min_up = idx_sort[L:][idx_sort[L:] < L]
                if len(idx_min_low) > 0:
                    for bb in range(len(idx_min_low)):
                        low = int(idx_min_low[bb])
                        up = int(idx_min_up[bb])
                        llrs_list[up] = np.copy(llrs_list[low])
                        s_list[up] = np.copy(s_list[low])
                        s_list[up][0, ii] = 1 - s_list[low][0, ii]
                    PM[idx_min_up] = PM_DM[idx_min_low + L]

    best = int(np.argmin(PM))
    return s_list[best][0, :].copy(), float(PM[best])


if _HAS_NUMBA:
    _f_node_minsum = numba.jit(nopython=True)(_f_node_minsum)
    _g_node = numba.jit(nopython=True)(_g_node)
    _B_check = numba.jit(nopython=True)(_B_check)
    _s_updater = numba.jit(nopython=True)(_s_updater)
    _Li = numba.jit(nopython=True)(_Li)
    sc_decoder_impl = numba.jit(nopython=True)(sc_decoder_impl)
    scl_decoder_impl = numba.jit(nopython=True)(scl_decoder_impl)
