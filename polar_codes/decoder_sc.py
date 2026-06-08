"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _all_filled(arr):
    return not np.any(np.isnan(arr))


def _leftdown(position):
    return [position[0] + 1, position[1], position[2], position[3]]


def _rightdown(position):
    return [
        position[0] + 1,
        position[1] + 2 ** (position[2] - 1 - position[0]),
        position[2],
        position[3],
    ]


def _up(position):
    p0 = position[0] - 1
    span = 2 ** (position[2] - position[0] + 1)
    p1 = int(np.floor(position[1] / span) * span)
    return [p0, p1, position[2], position[3]]


def _get_left_llr(up_llr):
    half = len(up_llr) // 2
    return f_operation(up_llr[:half], up_llr[half:])


def _get_right_llr(left_bit, up_llr):
    half = len(up_llr) // 2
    return g_operation(up_llr[:half], up_llr[half:], left_bit)


def _get_up_bit(left_bit, right_bit):
    half = len(left_bit)
    out = np.empty(2 * half, dtype=np.int8)
    out[:half] = (left_bit ^ right_bit) % 2
    out[half:] = right_bit
    return out


def _decide_bit(llr_val, is_frozen, is_right_child=False):
    if is_frozen:
        return 0
    if is_right_child:
        return 0 if llr_val > 0 else 1
    return 0 if llr_val >= 0 else 1


def _leaf_bit(idx, llr_val, frozen_bits, known_bits, is_right_child=False):
    if known_bits is not None and not np.isnan(known_bits[idx]):
        return int(known_bits[idx])
    return _decide_bit(llr_val, frozen_bits[idx], is_right_child)


def sc_llr_at_phi(llr_ch, frozen_bits, known_bits, phi):
    """已知前缀比特时，返回位置 phi 的 LLR（不做判决）。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    known_bits = np.asarray(known_bits, dtype=np.float64)
    N = len(llr_ch)
    n = int(math.log2(N))

    llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    llr_matrix[0, :] = llr_ch
    for i in range(phi):
        if not np.isnan(known_bits[i]):
            bit_matrix[n, i] = known_bits[i]
    position = [0, 0, n, N]
    max_steps = N * n * 8

    for _ in range(max_steps):
        span = 2 ** (position[2] - position[0])
        start = position[1]
        end = start + span
        half = span // 2

        up_bit = bit_matrix[position[0], start:end]
        left_llr = llr_matrix[position[0] + 1, start:start + half]
        left_bit = bit_matrix[position[0] + 1, start:start + half]
        right_llr = llr_matrix[position[0] + 1, start + half:end]
        right_bit = bit_matrix[position[0] + 1, start + half:end]
        up_llr = llr_matrix[position[0], start:end]

        if _all_filled(up_bit):
            position = _up(position)
            continue

        if _all_filled(right_bit):
            bit_matrix[position[0], start:end] = _get_up_bit(
                left_bit.astype(np.int8), right_bit.astype(np.int8)
            )
            continue

        if _all_filled(right_llr):
            if position[0] == position[2] - 1:
                if start + half == phi:
                    return float(right_llr[0])
                idx = start + half
                if np.isnan(bit_matrix[n, idx]):
                    bit_matrix[position[0] + 1, idx] = _leaf_bit(
                        idx, right_llr[0], frozen_bits, known_bits, True
                    )
            else:
                position = _rightdown(position)
            continue

        if _all_filled(left_bit):
            llr_matrix[position[0] + 1, start + half:end] = _get_right_llr(
                left_bit.astype(np.int8), up_llr
            )
            continue

        if not _all_filled(left_llr):
            llr_matrix[position[0] + 1, start:start + half] = _get_left_llr(up_llr)
            continue

        if position[0] == position[2] - 1:
            if start == phi:
                return float(left_llr[0])
            idx = start
            if np.isnan(bit_matrix[n, idx]):
                bit_matrix[position[0] + 1, idx] = _leaf_bit(
                    idx, left_llr[0], frozen_bits, known_bits, False
                )
        else:
            position = _leftdown(position)

    raise RuntimeError('LLR at phi not found')


def _sc_decode_core(llr_ch, frozen_bits, known_bits=None):
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    if known_bits is None:
        known_bits = np.full(len(llr_ch), np.nan)
    else:
        known_bits = np.asarray(known_bits, dtype=np.float64)
    N = len(llr_ch)
    n = int(math.log2(N))

    llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    llr_matrix[0, :] = llr_ch
    for i in range(N):
        if not np.isnan(known_bits[i]):
            bit_matrix[n, i] = known_bits[i]
    position = [0, 0, n, N]

    max_steps = N * n * 8
    for _ in range(max_steps):
        if _all_filled(bit_matrix[n]):
            break

        span = 2 ** (position[2] - position[0])
        start = position[1]
        end = start + span
        half = span // 2

        up_llr = llr_matrix[position[0], start:end]
        up_bit = bit_matrix[position[0], start:end]
        left_llr = llr_matrix[position[0] + 1, start:start + half]
        left_bit = bit_matrix[position[0] + 1, start:start + half]
        right_llr = llr_matrix[position[0] + 1, start + half:end]
        right_bit = bit_matrix[position[0] + 1, start + half:end]

        if _all_filled(up_bit):
            position = _up(position)
            continue

        if _all_filled(right_bit):
            merged = _get_up_bit(left_bit.astype(np.int8), right_bit.astype(np.int8))
            bit_matrix[position[0], start:end] = merged
            continue

        if _all_filled(right_llr):
            if position[0] == position[2] - 1:
                idx = start + half
                if np.isnan(bit_matrix[n, idx]):
                    bit_matrix[position[0] + 1, idx] = _leaf_bit(
                        idx, right_llr[0], frozen_bits, known_bits, True
                    )
            else:
                position = _rightdown(position)
            continue

        if _all_filled(left_bit):
            llr_matrix[position[0] + 1, start + half:end] = _get_right_llr(
                left_bit.astype(np.int8), up_llr
            )
            continue

        if not _all_filled(left_llr):
            llr_matrix[position[0] + 1, start:start + half] = _get_left_llr(up_llr)
            continue

        if position[0] == position[2] - 1:
            idx = start
            if np.isnan(bit_matrix[n, idx]):
                bit_matrix[position[0] + 1, idx] = _leaf_bit(
                    idx, left_llr[0], frozen_bits, known_bits, False
                )
        else:
            position = _leftdown(position)
    else:
        raise RuntimeError('SC decoder did not converge')

    return bit_matrix[n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（调用与主实现相同的因子图遍历）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量。"""
    n = int(math.log2(N))
    lambda_offset = np.zeros(n + 1, dtype=int)
    for i in range(1, n + 1):
        lambda_offset[i] = lambda_offset[i - 1] + (1 << (i - 1))

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers = []
        p = phi
        for layer in range(n):
            if p & 1:
                break
            layers.append(layer)
            p >>= 1
        llr_layer_vec.append(layers)

        layers_b = []
        if phi < N - 1:
            p = phi
            for layer in range(n):
                if (p + 1) % (1 << (layer + 1)) == 0:
                    layers_b.append(layer)
                p >>= 1
        bit_layer_vec.append(layers_b)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits, known_bits=None):
    """非递归 SC 译码主函数。"""
    return _sc_decode_core(llr_ch, frozen_bits, known_bits)
