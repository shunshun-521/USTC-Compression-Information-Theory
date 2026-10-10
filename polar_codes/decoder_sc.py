"""
极化码 SC（串行抵消）译码器
非递归 lazy LLR 计算（与蝶形编码器 polar_encode 配套）
"""
import numpy as np

try:
    from numba import jit

    _NUMBA = True
except ImportError:
    _NUMBA = False

    def jit(*args, **kwargs):
        def wrap(fn):
            return fn

        return wrap


@jit(nopython=True)
def _f_node_minsum(a, b):
    return np.sign(a * b) * np.minimum(np.abs(a), np.abs(b))


@jit(nopython=True)
def _g_node(LLR1, LLR2, s):
    return LLR1 * (1 - 2 * s) + LLR2


@jit(nopython=True)
def _B_check(ll, ii):
    return (ii // (1 << ll)) % 2


@jit(nopython=True)
def _s_updater(ll, ii, s):
    if _B_check(ll - 1, ii):
        s[ll, ii] = s[ll - 1, ii]
    else:
        if s[ll - 1, ii] == -1:
            _s_updater(ll - 1, ii, s)
        if s[ll - 1, ii + (1 << (ll - 1))] == -1:
            _s_updater(ll - 1, ii + (1 << (ll - 1)), s)
        s[ll, ii] = s[ll - 1, ii] ^ s[ll - 1, ii + (1 << (ll - 1))]


@jit(nopython=True)
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


@jit(nopython=True)
def _sc_decode_core(llr_channel, if_information_bit):
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
    return s[0, :]


def frozen_bits_to_info_mask(frozen_bits):
    """frozen_bits 中 1/True 为冻结位 -> 信息位掩码 1=信息"""
    fb = np.asarray(frozen_bits, dtype=np.int8)
    return (1 - fb).astype(np.int8)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主入口"""
    llr_ch = np.asarray(llr_ch, dtype=np.float32)
    info_mask = frozen_bits_to_info_mask(frozen_bits)
    return _sc_decode_core(llr_ch, info_mask).astype(int)


def sc_decode_recursive(llr_ch, frozen_bits):
    """与 sc_decode 相同（接口兼容）"""
    return sc_decode(llr_ch, frozen_bits)


def precompute_sc_indices(N):
    """接口占位（lazy SC 不需要预计算）"""
    n = int(np.log2(N))
    return [1 << i for i in range(n + 1)], [[] for _ in range(N)], [[] for _ in range(N)]


def f_operation(La, Lb):
    return _f_node_minsum(La, Lb)


def g_operation(La, Lb, u_hat):
    return _g_node(La, Lb, u_hat)
