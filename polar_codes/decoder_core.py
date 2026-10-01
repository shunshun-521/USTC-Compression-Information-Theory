"""
极化码 SC/SCL 核心（惰性 LLR 更新，min-sum f 节点）
改编自公开教学实现，与 B_N F^{\\otimes n} 编码 + 信道 LLR 比特倒序对齐。
"""
import numpy as np
import numba


@numba.jit(nopython=True)
def _f_node_minsum(a, b):
    return np.sign(a * b) * np.minimum(np.abs(a), np.abs(b))


@numba.jit(nopython=True)
def _g_node(llr1, llr2, s):
    return llr1 * (1 - 2 * s) + llr2


@numba.jit(nopython=True)
def _b_check(ll, ii):
    return (ii // (1 << ll)) % 2


@numba.jit(nopython=True)
def _s_updater(ll, ii, s):
    if _b_check(ll - 1, ii):
        s[ll, ii] = s[ll - 1, ii]
    else:
        if s[ll - 1, ii] == -1:
            _s_updater(ll - 1, ii, s)
        if s[ll - 1, ii + (1 << (ll - 1))] == -1:
            _s_updater(ll - 1, ii + (1 << (ll - 1)), s)
        s[ll, ii] = s[ll - 1, ii] ^ s[ll - 1, ii + (1 << (ll - 1))]


@numba.jit(nopython=True)
def _li(ll, ii, llrs, s):
    if llrs[ll, ii] != -np.inf:
        return llrs[ll, ii]
    if _b_check(ll, ii) == 0:
        llrs[ll, ii] = _f_node_minsum(
            _li(ll + 1, ii, llrs, s), _li(ll + 1, ii + (1 << ll), llrs, s)
        )
        return llrs[ll, ii]
    if ll > 0:
        _s_updater(ll, ii - (1 << ll), s)
    llrs[ll, ii] = _g_node(
        _li(ll + 1, ii - (1 << ll), llrs, s),
        _li(ll + 1, ii, llrs, s),
        s[ll, ii - (1 << ll)],
    )
    return llrs[ll, ii]


@numba.jit(nopython=True)
def sc_decode_core(llr_channel, if_information_bit):
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
            llrs[0, ii] = _li(0, ii, llrs, s)
            s[0, ii] = 1 if llrs[0, ii] < 0 else 0
    return s[0, :]


@numba.jit(nopython=True)
def scl_decode_core(llr_channel, if_information_bit, list_size):
    N = len(llr_channel)
    n = int(np.log2(N))
    L = list_size
    llrs_base = -np.inf * np.ones((n + 1, 1 << n), dtype=np.float32)
    llrs_base[-1, :] = llr_channel
    llrs = [llrs_base.copy() for _ in range(L)]
    s_base = -1 * np.ones((n + 1, 1 << n), dtype=np.int8)
    s = [s_base.copy() for _ in range(L)]
    dm = np.zeros(L)
    pm = np.inf * np.ones(L, dtype=np.float32)
    pm[0] = 0.0
    pm_dm = np.zeros(2 * L)

    for ii in range(N):
        if if_information_bit[ii] == 0:
            for dd in range(L):
                llrs[dd][0, ii] = _li(0, ii, llrs[dd], s[dd])
                s[dd][0, ii] = 0
                pm[dd] += -llrs[dd][0, ii] * (llrs[dd][0, ii] < 0)
        else:
            for dd in range(L):
                llrs[dd][0, ii] = _li(0, ii, llrs[dd], s[dd])
                s[dd][0, ii] = 1 if llrs[dd][0, ii] < 0 else 0
                dm[dd] = np.abs(llrs[dd][0, ii])

        if if_information_bit[ii] == 1 and L > 1:
            pm_dm[:L] = pm
            pm_dm[L:] = pm + dm
            idx_sort = np.argsort(pm_dm)
            idx_min_low = idx_sort[:L][idx_sort[:L] >= L] - L
            idx_min_up = idx_sort[L:][idx_sort[L:] < L]
            n_change = len(idx_min_low)
            if n_change != 0:
                for bb in range(n_change):
                    llrs[idx_min_up[bb]] = np.copy(llrs[idx_min_low[bb]])
                    s[idx_min_up[bb]] = np.copy(s[idx_min_low[bb]])
                    s[idx_min_up[bb]][0, ii] = 1 - s[idx_min_low[bb]][0, ii]
                pm[idx_min_up] = pm_dm[idx_min_low + L]

    dd_best = int(np.argmin(pm))
    return s[dd_best][0, :], pm[dd_best]
