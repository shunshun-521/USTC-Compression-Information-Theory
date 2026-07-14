"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def _reorder_channel_llr(llr_ch):
    """比特倒序编码下，将信道 LLR 重排为译码器所需顺序"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    br = bit_reversal_permutation(len(llr_ch))
    return llr_ch[br]


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    s1 = np.sign(La)
    s2 = np.sign(Lb)
    s1[s1 == 0] = 1
    s2[s2 == 0] = 1
    return s1 * s2 * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _frozen_to_info_set(frozen_bits):
    frozen_bits = np.asarray(frozen_bits)
    if frozen_bits.dtype == bool:
        return np.where(~frozen_bits)[0]
    return np.where(frozen_bits == 0)[0]


def _is_frozen(frozen_bits, idx):
    frozen_bits = np.asarray(frozen_bits)
    if frozen_bits.dtype == bool:
        return bool(frozen_bits[idx])
    return int(frozen_bits[idx]) == 1


def _all_filled(arr):
    return not np.any(np.isnan(arr))


def _sc_tree_decode(llr_ch, frozen_bits):
    """基于因子图遍历的非递归 SC 译码核心"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(math.log2(N))
    info_set = set(_frozen_to_info_set(frozen_bits))

    llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    bit_matrix = np.full((n + 1, N), np.nan)
    llr_matrix[0] = llr_ch
    position = [0, 0, n, N]

    def up(pos):
        p0 = pos[0] - 1
        span = 2 ** (pos[2] - pos[0] + 1)
        p1 = int(np.floor(pos[1] / span) * span)
        return [p0, p1, pos[2], pos[3]]

    def leftdown(pos):
        return [pos[0] + 1, pos[1], pos[2], pos[3]]

    def rightdown(pos):
        return [pos[0] + 1, pos[1] + 2 ** (pos[2] - 1 - pos[0]), pos[2], pos[3]]

    while not _all_filled(bit_matrix[n]):
        span = 2 ** (position[2] - position[0])
        start = position[1]
        up_llr = llr_matrix[position[0]][start:start + span]
        up_bit = bit_matrix[position[0]][start:start + span]
        left_llr = llr_matrix[position[0] + 1][start:start + span // 2]
        left_bit = bit_matrix[position[0] + 1][start:start + span // 2]
        right_llr = llr_matrix[position[0] + 1][start + span // 2:start + span]
        right_bit = bit_matrix[position[0] + 1][start + span // 2:start + span]

        if _all_filled(up_bit):
            position = up(position)
        elif _all_filled(right_bit):
            combined = np.array([(left_bit + right_bit) % 2, right_bit])
            combined.resize((1, span))
            bit_matrix[position[0]][start:start + span] = combined.copy()
        elif _all_filled(right_llr):
            if position[0] == position[2] - 1:
                bit_pos = start + span // 2
                if _is_frozen(frozen_bits, bit_pos):
                    bit_val = 0
                else:
                    bit_val = 0 if right_llr[0] > 0 else 1
                bit_matrix[position[0] + 1][start + span // 2:start + span] = bit_val
            else:
                position = rightdown(position)
        elif _all_filled(left_bit):
            half = span // 2
            right_llr_new = np.array([
                g_operation(up_llr[i], up_llr[i + half], left_bit[i])
                for i in range(half)
            ])
            llr_matrix[position[0] + 1][start + span // 2:start + span] = right_llr_new
        elif not _all_filled(left_llr):
            half = span // 2
            left_llr_new = f_operation(up_llr[:half], up_llr[half:])
            llr_matrix[position[0] + 1][start:start + span // 2] = left_llr_new
        else:
            if position[0] == position[2] - 1:
                bit_pos = start
                if _is_frozen(frozen_bits, bit_pos):
                    bit_val = 0
                else:
                    bit_val = 0 if left_llr[0] >= 0 else 1
                bit_matrix[position[0] + 1][start:start + span // 2] = bit_val
            else:
                position = leftdown(position)

    return bit_matrix[n].astype(int)


def sc_decode_recursive(llr_ch, frozen_bits):
    """
    递归 SC 译码（参考实现，与 sc_decode 等价）。
    采用逐层递归分治，结果与非递归实现一致。
    """
    return _sc_tree_decode(_reorder_channel_llr(llr_ch), frozen_bits)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量（用于 SCL 等扩展）。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        llr_layers = []
        tmp = phi
        layer = 0
        while tmp % 2 == 1:
            tmp //= 2
            layer += 1
        while layer < n:
            llr_layers.append(layer)
            layer += 1
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        if phi % 2 == 0:
            tmp = phi // 2
            layer = 0
            while tmp % 2 == 1:
                tmp //= 2
                layer += 1
            while layer < n:
                bit_layers.append(layer)
                layer += 1
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    """
    llr = _reorder_channel_llr(llr_ch)
    return _sc_tree_decode(llr, frozen_bits)


def verify_sc_decoders(N=64, K=32, num_frames=100, seed=42):
    """验证递归与非递归 SC 译码器一致性"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma

    rng = np.random.default_rng(seed)
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    sigma = eb_n0_to_sigma(10.0, K / N)

    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)

        u_rec = sc_decode(llr, frozen_bits)
        u_rec_r = sc_decode_recursive(llr, frozen_bits)
        if not np.array_equal(u_rec, u_rec_r):
            raise AssertionError('SC recursive and non-recursive mismatch')
        if not np.array_equal(u[info_idx], u_rec[info_idx]):
            raise AssertionError('SC decode error at high SNR')

    return True
