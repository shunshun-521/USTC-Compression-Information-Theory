"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _all_filled,
    _decide_bit,
    _get_left_llr,
    _get_right_llr,
    _get_up_bit,
    _leftdown,
    _prepare_channel_llr,
    _rightdown,
    _up,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc8_bits(info_bits):
    poly = CRC8_POLY
    reg = 0
    for bit in np.asarray(info_bits, dtype=int).flatten():
        reg ^= int(bit) << 7
        for _ in range(8):
            if reg & 0x80:
                reg = ((reg << 1) ^ poly) & 0xFF
            else:
                reg = (reg << 1) & 0xFF
    return [(reg >> (7 - i)) & 1 for i in range(8)]


def _crc16_bits(info_bits):
    poly = CRC16_POLY
    reg = 0
    for bit in np.asarray(info_bits, dtype=int).flatten():
        reg ^= int(bit) << 15
        for _ in range(16):
            if reg & 0x8000:
                reg = ((reg << 1) ^ poly) & 0xFFFF
            else:
                reg = (reg << 1) & 0xFFFF
    return [(reg >> (15 - i)) & 1 for i in range(16)]


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int).flatten()
    if crc_length == 8:
        crc_bits = _crc8_bits(info_bits)
    elif crc_length == 16:
        crc_bits = _crc16_bits(info_bits)
    else:
        raise ValueError(f"Unsupported CRC length: {crc_length}")
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int).flatten()
    if len(bits) < crc_length:
        return False
    info = bits[:-crc_length]
    recv = bits[-crc_length:]
    if crc_length == 8:
        expected = _crc8_bits(info)
    elif crc_length == 16:
        expected = _crc16_bits(info)
    else:
        raise ValueError(f"Unsupported CRC length: {crc_length}")
    return np.array_equal(recv, expected)


def _get_up_loc(bit_matrix):
    N = bit_matrix.shape[1]
    n = int(math.log2(N))
    detect_array = bit_matrix[n]
    detect = -1
    for i in range(N):
        if detect_array[i] not in (0, 1):
            detect = i - 1
            break
    if detect == -1:
        return [0, 0]
    if detect % 2 == 0:
        return [n - 1, detect]
    return [n - 1, detect - 1]


def _sc_step_to_phi(llr_matrix, bit_matrix, info_set, split_pos, frozen_val=0):
    """运行 SC 状态机直到完成 split_pos 处比特判决。"""
    N = bit_matrix.shape[1]
    n = int(math.log2(N))
    loc = _get_up_loc(bit_matrix)
    position = [loc[0], loc[1], n, N]

    while bit_matrix[n][split_pos] not in (0, 1):
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
            up_bit_val = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0]][position[1] : position[1] + span] = up_bit_val.flatten()
        elif _all_filled(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                right_bit_val = _decide_bit(
                    right_llr[0], right_bit_pos, info_set, frozen_val
                )
                bit_matrix[position[0] + 1][position[1] + half] = right_bit_val
            else:
                position = _rightdown(position)
        elif _all_filled(left_bit):
            right_llr_val = _get_right_llr(left_bit, up_llr)
            llr_matrix[position[0] + 1][position[1] + half : position[1] + span] = (
                right_llr_val
            )
        elif not _all_filled(left_llr):
            left_llr_val = _get_left_llr(up_llr)
            llr_matrix[position[0] + 1][position[1] : position[1] + half] = left_llr_val
        else:
            if position[0] == position[2] - 1:
                left_bit_pos = position[1]
                left_bit_val = _decide_bit(
                    left_llr[0], left_bit_pos, info_set, frozen_val
                )
                bit_matrix[position[0] + 1][position[1]] = left_bit_val
            else:
                position = _leftdown(position)

    return llr_matrix, bit_matrix


def _path_metric_add(llr_val, u):
    hard = 0 if llr_val >= 0 else 1
    return 0.0 if u == hard else abs(llr_val)


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy：路径分裂时复制矩阵）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.info_indices = np.where(self.frozen_bits == 0)[0]
        self.info_set = set(int(i) for i in self.info_indices)
        self.list_size = list_size
        self.crc_length = crc_length

    def _new_path(self, y_llr):
        llr_matrix = np.full((self.n + 1, self.N), np.nan)
        bit_matrix = np.full((self.n + 1, self.N), np.nan)
        llr_matrix[0] = y_llr
        return llr_matrix, bit_matrix

    def decode(self, llr_ch):
        y_llr = _prepare_channel_llr(llr_ch, self.N)
        llr_m, bit_m = self._new_path(y_llr)
        paths = [(llr_m, bit_m, 0.0)]

        for phi in range(self.N):
            new_paths = []
            for llr_matrix, bit_matrix, pm in paths:
                llr_matrix, bit_matrix = _sc_step_to_phi(
                    llr_matrix, bit_matrix, self.info_set, phi
                )
                leaf_llr = llr_matrix[self.n][phi]

                if phi not in self.info_set:
                    u = 0
                    new_pm = pm + _path_metric_add(leaf_llr, u)
                    bit_matrix[self.n][phi] = u
                    new_paths.append((llr_matrix, bit_matrix, new_pm))
                else:
                    for u in (0, 1):
                        llr_copy = llr_matrix.copy()
                        bit_copy = bit_matrix.copy()
                        bit_copy[self.n][phi] = u
                        new_pm = pm + _path_metric_add(leaf_llr, u)
                        new_paths.append((llr_copy, bit_copy, new_pm))

            new_paths.sort(key=lambda x: x[2])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            for llr_matrix, bit_matrix, pm in paths:
                u_hat = bit_matrix[self.n].astype(int)
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    return u_hat, pm

        best = min(paths, key=lambda x: x[2])
        return best[1][self.n].astype(int), best[2]
