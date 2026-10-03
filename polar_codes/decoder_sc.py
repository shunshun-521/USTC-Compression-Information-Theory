"""
极化码 SC（串行抵消）译码器
基于惰性 LLR 更新的非递归实现（与标准极化因子图一致）
"""
import math
import numpy as np

try:
    from numba import jit

    _NUMBA = True
except ImportError:  # pragma: no cover
    _NUMBA = False

    def jit(*_args, **_kwargs):
        def wrapper(fn):
            return fn

        return wrapper


@jit(nopython=True)
def _f_node_minsum(a, b):
    return np.sign(a * b) * min(abs(a), abs(b))


@jit(nopython=True)
def _g_node(llr1, llr2, s):
    return llr1 * (1 - 2 * s) + llr2


@jit(nopython=True)
def _b_check(ll, ii):
    return (ii // (1 << ll)) % 2


@jit(nopython=True)
def _s_updater(ll, ii, s):
    if _b_check(ll - 1, ii):
        s[ll, ii] = s[ll - 1, ii]
    else:
        if s[ll - 1, ii] == -1:
            _s_updater(ll - 1, ii, s)
        if s[ll - 1, ii + (1 << (ll - 1))] == -1:
            _s_updater(ll - 1, ii + (1 << (ll - 1)), s)
        s[ll, ii] = s[ll - 1, ii] ^ s[ll - 1, ii + (1 << (ll - 1))]


@jit(nopython=True)
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


@jit(nopython=True)
def _sc_decode_core(llr_channel, if_information_bit):
    n = int(math.log2(len(llr_channel)))
    llrs = -np.inf * np.ones((n + 1, 1 << n), dtype=np.float64)
    llrs[-1, :] = llr_channel
    s = -1 * np.ones((n + 1, 1 << n), dtype=np.int8)
    for ii in range(1 << n):
        if if_information_bit[ii] == 0:
            s[0, ii] = 0
            llrs[0, ii] = np.inf
        else:
            llrs[0, ii] = _li(0, ii, llrs, s)
            s[0, ii] = 1 if llrs[0, ii] < 0 else 0
    return s[0, :]


def f_operation(La, Lb):
    """对外暴露的 f 运算（min-sum），供 BP 等模块复用。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """对外暴露的 g 运算。"""
    return (1 - 2 * u_hat) * La + Lb


def precompute_sc_indices(N):
    """兼容 SCL 接口的占位预计算。"""
    n = int(math.log2(N))
    return [1 << i for i in range(n + 1)], [[] for _ in range(N)], [[] for _ in range(N)]


def _frozen_to_info_mask(frozen_bits):
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    if_info = np.zeros(len(frozen_bits), dtype=np.int8)
    if_info[~frozen_bits] = 1
    return if_info


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    if_info = _frozen_to_info_mask(frozen_bits)
    return _sc_decode_core(llr_ch.astype(np.float64), if_info).astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归别名（与 sc_decode 相同）。"""
    return sc_decode(llr, frozen_bits)
