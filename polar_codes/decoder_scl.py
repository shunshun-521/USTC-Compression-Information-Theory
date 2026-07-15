"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _all_filled,
    _frozen_to_info_set,
    _is_frozen,
    _reorder_channel_llr,
    _sc_tree_decode,
    f_operation,
    g_operation,
    precompute_sc_indices,
    sc_decode,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_process(bits, poly, crc_length):
    reg = 0
    width = 8 if crc_length == 8 else 16
    mask = (1 << width) - 1
    top = 1 << (width - 1)
    for bit in bits:
        reg ^= int(bit) << (width - 1)
        for _ in range(width):
            if reg & top:
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_process(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_process(bits, poly, crc_length) == 0


def _up(pos):
    p0 = pos[0] - 1
    span = 2 ** (pos[2] - pos[0] + 1)
    p1 = int(np.floor(pos[1] / span) * span)
    return [p0, p1, pos[2], pos[3]]


def _leftdown(pos):
    return [pos[0] + 1, pos[1], pos[2], pos[3]]


def _rightdown(pos):
    return [pos[0] + 1, pos[1] + 2 ** (pos[2] - 1 - pos[0]), pos[2], pos[3]]


def _get_up_loc(bit_matrix, n, N):
    detect = -1
    for i in range(N):
        if np.isnan(bit_matrix[n][i]):
            detect = i - 1
            break
    if detect == -1 and np.isnan(bit_matrix[n][0]):
        return [0, 0]
    if detect % 2 == 0:
        return [n - 1, max(detect, 0)]
    return [n - 1, max(detect - 1, 0)]


def _pm_update(llr_slice, bit_slice):
    pm = 0.0
    for llr_val, bit in zip(llr_slice, bit_slice):
        hard = 0 if llr_val >= 0 else 1
        if hard != int(bit):
            pm += abs(llr_val)
    return pm


def _sc_step_to_split(llr_matrix, bit_matrix, frozen_bits, split_pos):
    N = bit_matrix.shape[1]
    n = int(math.log2(N))
    loc = _get_up_loc(bit_matrix, n, N)
    position = [loc[0], loc[1], n, N]

    while np.isnan(bit_matrix[n][split_pos]):
        if position[0] < 0:
            break
        span = 2 ** (position[2] - position[0])
        start = position[1]
        up_llr = llr_matrix[position[0]][start:start + span]
        left_llr = llr_matrix[position[0] + 1][start:start + span // 2]
        left_bit = bit_matrix[position[0] + 1][start:start + span // 2]
        right_llr = llr_matrix[position[0] + 1][start + span // 2:start + span]
        right_bit = bit_matrix[position[0] + 1][start + span // 2:start + span]

        if _all_filled(bit_matrix[position[0]][start:start + span]):
            position = _up(position)
        elif _all_filled(right_bit):
            combined = np.array([(left_bit + right_bit) % 2, right_bit])
            combined.resize((1, span))
            bit_matrix[position[0]][start:start + span] = combined.copy()
        elif _all_filled(right_llr):
            if position[0] == position[2] - 1:
                bit_pos = start + span // 2
                if _is_frozen(frozen_bits, bit_pos):
                    bit_val = 0.0
                else:
                    bit_val = 0.0 if right_llr[0] > 0 else 1.0
                bit_matrix[position[0] + 1][start + span // 2:start + span] = bit_val
            else:
                position = _rightdown(position)
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
                    bit_val = 0.0
                else:
                    bit_val = 0.0 if left_llr[0] >= 0 else 1.0
                bit_matrix[position[0] + 1][start:start + span // 2] = bit_val
            else:
                position = _leftdown(position)

    return llr_matrix, bit_matrix


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits)
        self.info_indices = _frozen_to_info_set(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.lambda_offset, self.llr_layer_vec, self.bit_layer_vec = precompute_sc_indices(N)

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm"""
        llr_ch = _reorder_channel_llr(llr_ch)

        if self.list_size == 1:
            return _sc_tree_decode(llr_ch, self.frozen_bits), 0.0

        N = self.N
        n = self.n
        info_positions = list(self.info_indices)

        llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
        bit_matrix = np.full((n + 1, N), np.nan)
        llr_matrix[0] = llr_ch

        llr_list = [llr_matrix]
        bit_list = [bit_matrix]
        pm_list = [0.0]

        prev_split = -1
        for split_pos in info_positions:
            new_llr_list = []
            new_bit_list = []
            new_pm_list = []

            for llr_m, bit_m, pm in zip(llr_list, bit_list, pm_list):
                llr_m = llr_m.copy()
                bit_m = bit_m.copy()
                llr_m, bit_m = _sc_step_to_split(llr_m, bit_m, self.frozen_bits, split_pos)

                llr_slice = llr_m[n][prev_split + 1:split_pos + 1]
                bit_slice = bit_m[n][prev_split + 1:split_pos + 1]
                pm_add = _pm_update(llr_slice, bit_slice)

                new_llr_list.append(llr_m)
                new_bit_list.append(bit_m)
                new_pm_list.append(pm + pm_add)

                bit_wrong = bit_m.copy()
                bit_wrong[n][split_pos] = 1.0 - bit_wrong[n][split_pos]
                wrong_slice = bit_wrong[n][prev_split + 1:split_pos + 1]
                pm_wrong = pm + _pm_update(llr_slice, wrong_slice)

                new_llr_list.append(llr_m.copy())
                new_bit_list.append(bit_wrong)
                new_pm_list.append(pm_wrong)

            order = np.argsort(new_pm_list)[:self.list_size]
            llr_list = [new_llr_list[i] for i in order]
            bit_list = [new_bit_list[i] for i in order]
            pm_list = [new_pm_list[i] for i in order]
            prev_split = split_pos

        for i in range(len(llr_list)):
            llr_list[i], bit_list[i] = _sc_step_to_split(
                llr_list[i].copy(), bit_list[i].copy(), self.frozen_bits, N - 1,
            )
            pm_list[i] += _pm_update(
                llr_list[i][n][prev_split + 1:N],
                bit_list[i][n][prev_split + 1:N],
            )

        order = np.argsort(pm_list)

        if self.crc_length > 0:
            for idx in order:
                u_candidate = np.nan_to_num(bit_list[idx][n], nan=0.0).astype(int)
                if crc_check(u_candidate[self.info_indices], self.crc_length):
                    return u_candidate, pm_list[idx]
            best = order[0]
        else:
            best = order[0]

        u_hat = np.nan_to_num(bit_list[best][n], nan=0.0).astype(int)
        return u_hat, pm_list[best]


def verify_scl_equals_sc(N=64, K=32, seed=0):
    """验证 L=1 的 SCL 等价于 SC"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma

    rng = np.random.default_rng(seed)
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(5.0, K / N)

    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        llr = compute_llr(
            awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma,
        )
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            raise AssertionError('SCL L=1 != SC')

    return True
