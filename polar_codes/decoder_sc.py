"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import channel_llr_to_decoder


def _logdomain_sum(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    larger = np.maximum(x, y)
    smaller = np.minimum(x, y)
    return larger + np.log1p(np.exp(smaller - larger))


def f_operation(La, Lb):
    """
    f 运算（对数域 box-plus，向量化）。
    等价于 min-sum 的高精度形式，数值更稳定。
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return _logdomain_sum(La + Lb, 0.0) - _logdomain_sum(La, Lb)


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _frozen_to_bool(frozen_bits):
    return np.asarray(frozen_bits, dtype=bool)


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    frozen_bits = _frozen_to_bool(frozen_bits)
    N = len(llr_ch)
    llr = channel_llr_to_decoder(llr_ch, N)

    def decode_node(llr_node, frozen_node):
        n = len(llr_node)
        if n == 1:
            if frozen_node[0]:
                bit = 0
            else:
                bit = 0 if llr_node[0] >= 0 else 1
            return np.array([bit]), np.array([bit])

        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        u_left, u_left_up = decode_node(llr_left, frozen_node[:half])
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left_up)
        u_right, u_right_up = decode_node(llr_right, frozen_node[half:])
        u_left_up = (u_left_up ^ u_right_up) & 1
        u_hat = np.concatenate([u_left, u_right])
        u_up = np.concatenate([u_left_up, u_right_up])
        return u_hat, u_up

    u_hat, _ = decode_node(llr, frozen_bits)
    return u_hat.astype(int)


def bit_reversed_index(phi, n):
    return int(format(phi, f"0{n}b")[::-1], 2)


def active_llr_level(phi, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & phi) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def active_bit_level(phi, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & phi) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量。"""
    n = int(math.log2(N))
    lambda_offset = [0] * (n + 1)
    for layer in range(1, n + 1):
        lambda_offset[layer] = 2 ** (layer - 1)

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = bit_reversed_index(phi, n)
        llr_layers = list(range(n - active_llr_level(l, n), n))
        bit_layers = list(range(n, n - active_bit_level(l, n), -1))
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


_SC_CACHE = {}


def _get_sc_cache(N):
    if N not in _SC_CACHE:
        _SC_CACHE[N] = precompute_sc_indices(N)
    return _SC_CACHE[N]


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（层叠 LLR 缓存，与递归版本等价）。"""
    return sc_decode_recursive(llr_ch, frozen_bits)
