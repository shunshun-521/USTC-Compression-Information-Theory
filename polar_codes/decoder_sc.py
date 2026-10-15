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
    """
    递归 SC 译码（参考实现，用于校验非递归版本）。
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits)
    if frozen_bits.dtype == bool:
        frozen_mask = frozen_bits
    else:
        frozen_mask = frozen_bits != 0
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            if frozen_mask[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return

        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        decode_node(llr_left, bit_offset)
        u_left = u_hat[bit_offset : bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        decode_node(llr_right, bit_offset + half)

    import math

    decode_node(llr, 0)
    return u_hat


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码的层索引（Arıkan 蝶形图）。
    """
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers_llr = []
        p = phi
        while p % 2 == 1 and p < N:
            layers_llr.append(int(np.log2(p & -p)))
            p //= 2
        llr_layer_vec.append(layers_llr)
        layers_bit = []
        q = phi
        for layer in range(n):
            if (q >> layer) & 1:
                layers_bit.append(layer)
        bit_layer_vec.append(layers_bit)
    return lambda_offset, llr_layer_vec, bit_layer_vec
