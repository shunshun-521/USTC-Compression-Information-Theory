"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy

import numpy as np

from decoder_sc import (
    _all_filled,
    _frozen_to_info,
    _get_bit,
    _get_left_llr,
    _get_right_llr,
    _get_up_bit,
    _leftdown,
    _prepare_llr,
    _rightdown,
    _sc_tree_decode,
    _up,
    f_operation,
    g_operation,
    sc_decode,
)
from encoder import bit_reversal_permutation, polar_encode


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)

    crc_bits = np.array(
        [(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


def _get_up_loc(bit_matrix):
    N = bit_matrix.shape[1]
    n = int(np.log2(N))
    for i in range(N):
        if bit_matrix[n, i] == 0 or bit_matrix[n, i] == 1:
            if i == 0:
                return [0, 0]
            span = 1
            while i % (2 * span) != 0:
                span *= 2
            return [n - int(np.log2(span)), i - span]
    return [0, 0]


def _pm_update(llr_slice, bit_slice):
    penalty = 0.0
    for llr, bit in zip(llr_slice, bit_slice):
        hard = 0 if llr >= 0 else 1
        if hard != bit:
            penalty += abs(llr)
    return penalty


def _sc_step_to_split(llr_matrix, bit_matrix, info_set, frozen_val, split_pos):
    """译码至信息位 split_pos 判决完成。"""
    N = bit_matrix.shape[1]
    n = int(np.log2(N))
    loc = _get_up_loc(bit_matrix)
    position = [loc[0], loc[1], n, N]

    while bit_matrix[n, split_pos] != 0 and bit_matrix[n, split_pos] != 1:
        span = 2 ** (position[2] - position[0])
        up_llr = llr_matrix[position[0]][position[1] : position[1] + span]
        up_bit = bit_matrix[position[0]][position[1] : position[1] + span]
        half = span // 2
        left_llr = llr_matrix[position[0] + 1][position[1] : position[1] + half]
        left_bit = bit_matrix[position[0] + 1][position[1] : position[1] + half]
        right_llr = llr_matrix[position[0] + 1][position[1] + half : position[1] + span]
        right_bit = bit_matrix[position[0] + 1][position[1] + half : position[1] + span]

        if _all_filled(up_bit):
            position = _up(position)
        elif _all_filled(right_bit):
            bit_matrix[position[0]][position[1] : position[1] + span] = _get_up_bit(
                left_bit, right_bit
            )
        elif _all_filled(right_llr):
            if position[0] == position[2] - 1:
                right_pos = position[1] + 1
                bit_matrix[position[0] + 1][position[1] + half : position[1] + span] = _get_bit(
                    right_llr[0], right_pos in info_set, frozen_val
                )
            else:
                position = _rightdown(position)
        elif _all_filled(left_bit):
            llr_matrix[position[0] + 1][position[1] + half : position[1] + span] = _get_right_llr(
                left_bit, up_llr
            )
        elif not _all_filled(left_llr):
            llr_matrix[position[0] + 1][position[1] : position[1] + half] = _get_left_llr(up_llr)
        else:
            if position[0] == position[2] - 1:
                left_pos = position[1]
                bit_matrix[position[0] + 1][position[1] : position[1] + half] = _get_bit(
                    left_llr[0], left_pos in info_set, frozen_val
                )
            else:
                position = _leftdown(position)

    return llr_matrix, bit_matrix


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.info_positions = _frozen_to_info(self.frozen_bits)
        self.info_set = set(int(i) for i in self.info_positions)
        self.list_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        """主译码函数。"""
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr = _prepare_llr(llr_ch, self.N)
        n = self.n
        N = self.N

        llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
        bit_matrix = np.full((n + 1, N), np.nan)
        llr_matrix[0] = llr

        llr_list = [llr_matrix.copy()]
        bit_list = [bit_matrix.copy()]
        pm_list = [0.0]

        split_positions = [i for i in self.info_positions]
        for split_pos in split_positions:
            new_llr_list = []
            new_bit_list = []
            new_pm_list = []

            for llr_m, bit_m, pm in zip(llr_list, bit_list, pm_list):
                llr_m, bit_m = _sc_step_to_split(
                    llr_m.copy(), bit_m.copy(), self.info_set, 0, split_pos
                )
                llr_slice = llr_m[n][split_pos : split_pos + 1]
                for bit in (0, 1):
                    bit_copy = bit_m.copy()
                    bit_copy[n, split_pos] = bit
                    pm_new = pm + _pm_update(llr_slice, np.array([bit]))
                    new_llr_list.append(llr_m.copy())
                    new_bit_list.append(bit_copy)
                    new_pm_list.append(pm_new)

            order = np.argsort(new_pm_list)
            keep = order[: self.list_size]
            llr_list = [new_llr_list[i] for i in keep]
            bit_list = [new_bit_list[i] for i in keep]
            pm_list = [new_pm_list[i] for i in keep]

        for idx, (llr_m, bit_m, pm) in enumerate(zip(llr_list, bit_list, pm_list)):
            llr_m, bit_m = _sc_step_to_split(
                llr_m.copy(), bit_m.copy(), self.info_set, 0, N - 1
            )
            llr_list[idx] = llr_m
            bit_list[idx] = bit_m

        if self.crc_length > 0:
            valid = []
            for bit_m, pm in zip(bit_list, pm_list):
                u_hat = bit_m[n].astype(int)
                if crc_check(u_hat, self.crc_length):
                    valid.append((pm, u_hat))
            if valid:
                best_pm, u_hat = min(valid, key=lambda x: x[0])
                return u_hat, best_pm

        best_idx = int(np.argmin(pm_list))
        u_hat = bit_list[best_idx][n].astype(int)
        return u_hat, pm_list[best_idx]


if __name__ == "__main__":
    from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
    from construction import ga_construction
    from encoder import polar_encode

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(5.0, K / N)
    mismatches = 0
    for _ in range(20):
        u_sent = np.zeros(N, dtype=int)
        u_sent[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u_sent)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    assert mismatches == 0, f"SCL L=1 != SC: {mismatches}"
    print("SCL L=1 equivalence test passed")
