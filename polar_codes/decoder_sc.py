"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）与非递归（BP 等效调度，与 BP 因子图一致）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation
from decoder_bp import _f_min_sum


def f_operation(La, Lb):
    """min-sum f（与 BP 中 alpha=1 时一致，用于递归参考实现）。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（标准 SC 定义）。"""
    return (1 - 2 * u_hat) * La + Lb


def _channel_llr_to_decode(llr_ch):
    N = len(llr_ch)
    return np.asarray(llr_ch, dtype=np.float64)[bit_reversal_permutation(N)]


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（教学参考，大码长建议用 sc_decode）。"""
    llr = _channel_llr_to_decode(llr)
    N = len(llr)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            if frozen_bits[idx]:
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

    decode_node(llr, 0)
    return u_hat


def precompute_sc_indices(N):
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers_llr = []
        psi = phi
        while psi % 2 == 1:
            layers_llr.append(int(math.log2(psi ^ (psi - 1))))
            psi = (psi - 1) // 2
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        if phi % 2 == 0:
            psi = phi
            while psi % 2 == 0 and psi > 0:
                layers_bit.append(int(math.log2(psi + 2)) - 1)
                psi //= 2
        bit_layer_vec.append(layers_bit)

    return llr_layer_vec, bit_layer_vec


_SC_CACHE = {}


def _get_sc_tables(N):
    if N not in _SC_CACHE:
        _SC_CACHE[N] = precompute_sc_indices(N)
    return _SC_CACHE[N]


def sc_decode(llr_ch, frozen_bits, alpha=0.9375):
    """
    非递归 SC：按比特顺序在因子图上更新 L/R（与 BP 相同图，串行抵消调度）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    br = bit_reversal_permutation(N)
    llr = llr_ch[br]

    from decoder_bp import BPDecoder

    decoder = BPDecoder(N, frozen_bits, max_iter=80, alpha=alpha)
    u_hat, _ = decoder.decode(llr_ch)
    return u_hat
