"""
极化码 SC（串行抵消）译码器
提供递归接口与非递归（按需计算 LLR）实现，与 encoder.polar_encode 配套
"""
import numpy as np

try:
    from numba import jit

    _NUMBA = True
except ImportError:
    _NUMBA = False

    def jit(*_args, **_kwargs):
        def wrap(fn):
            return fn

        return wrap


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    u_hat = np.asarray(u_hat)
    return (1.0 - 2.0 * u_hat) * La + Lb


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
def _li(ll, ii, llrs, s, n):
    if llrs[ll, ii] != -np.inf:
        return llrs[ll, ii]
    if _b_check(ll, ii) == 0:
        llrs[ll, ii] = _f_node_minsum(
            _li(ll + 1, ii, llrs, s, n), _li(ll + 1, ii + (1 << ll), llrs, s, n)
        )
        return llrs[ll, ii]
    if ll > 0:
        _s_updater(ll, ii - (1 << ll), s)
    llrs[ll, ii] = _g_node(
        _li(ll + 1, ii - (1 << ll), llrs, s, n),
        _li(ll + 1, ii, llrs, s, n),
        s[ll, ii - (1 << ll)],
    )
    return llrs[ll, ii]


@jit(nopython=True)
def _sc_decode_core(llr_channel, frozen_mask, n):
    N = len(llr_channel)
    llrs = -np.inf * np.ones((n + 1, N), dtype=np.float64)
    llrs[n, :] = llr_channel
    s = -np.ones((n + 1, N), dtype=np.int8)
    u_hat = np.zeros(N, dtype=np.int8)
    for ii in range(N):
        if frozen_mask[ii]:
            s[0, ii] = 0
            llrs[0, ii] = np.inf
            u_hat[ii] = 0
        else:
            llrs[0, ii] = _li(0, ii, llrs, s, n)
            bit = 1 if llrs[0, ii] < 0 else 0
            s[0, ii] = bit
            u_hat[ii] = bit
    return u_hat


def _frozen_mask_from_bits(frozen_bits):
    fb = np.asarray(frozen_bits)
    if fb.dtype == bool:
        return fb.astype(np.int8)
    return (fb != 0).astype(np.int8)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（按需计算 LLR）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(np.log2(N))
    frozen_mask = _frozen_mask_from_bits(frozen_bits)
    u_hat = _sc_decode_core(llr_ch, frozen_mask, n)
    return u_hat.astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（与 sc_decode 相同实现）"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """文档接口：返回占位预计算表"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    return lambda_offset, [[] for _ in range(N)], [[] for _ in range(N)]
