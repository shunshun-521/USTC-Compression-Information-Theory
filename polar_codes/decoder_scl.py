"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import copy
import numpy as np
from decoder_sc import f_operation, g_operation, _all_ready, _left_down, _right_down, _up, _get_up_bit, _decide_bit

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if crc_length == 8:
                if reg & 0x80:
                    reg = ((reg << 1) ^ poly) & 0xFF
                else:
                    reg = (reg << 1) & 0xFF
            else:
                if reg & 0x8000:
                    reg = ((reg << 1) ^ poly) & 0xFFFF
                else:
                    reg = (reg << 1) & 0xFFFF
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if crc_length == 8:
                if reg & 0x80:
                    reg = ((reg << 1) ^ poly) & 0xFF
                else:
                    reg = (reg << 1) & 0xFF
            else:
                if reg & 0x8000:
                    reg = ((reg << 1) ^ poly) & 0xFFFF
                else:
                    reg = (reg << 1) & 0xFFFF
    return reg == 0


def _init_matrices(llr_ch, n, N):
    llr_matrix = np.ones((n + 1, N), dtype=np.float64)
    llr_matrix[llr_matrix == 1] = np.nan
    bit_matrix = llr_matrix.copy()
    llr_matrix[0] = llr_ch
    return llr_matrix, bit_matrix


def _get_up_loc(bit_matrix, n, N):
    for i in range(n + 1):
        for j in range(N):
            if np.isnan(bit_matrix[i, j]):
                return [i, j]
    return [0, 0]


def _sc_step_to_split(llr_matrix, bit_matrix, info_set, split_pos, n, N):
    """SC 译码至完成 split_pos 比特判决。"""
    loc = _get_up_loc(bit_matrix, n, N)
    position = [loc[0], loc[1], n, N]

    while not (
        bit_matrix[n, split_pos] == 0 or bit_matrix[n, split_pos] == 1
    ):
        span = 2 ** (position[2] - position[0])
        up_llr = llr_matrix[position[0], position[1] : position[1] + span]
        left_bit = bit_matrix[position[0] + 1, position[1] : position[1] + span // 2]
        right_bit = bit_matrix[
            position[0] + 1, position[1] + span // 2 : position[1] + span
        ]
        left_llr = llr_matrix[position[0] + 1, position[1] : position[1] + span // 2]
        right_llr = llr_matrix[
            position[0] + 1, position[1] + span // 2 : position[1] + span
        ]
        half = span // 2

        if _all_ready(bit_matrix[position[0], position[1] : position[1] + span]):
            position = _up(position)
        elif _all_ready(right_bit):
            up_bit_val = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0], position[1] : position[1] + span] = up_bit_val.copy()
        elif _all_ready(right_llr):
            if position[0] == position[2] - 1:
                rp = position[1] + 1
                bit_matrix[position[0] + 1, position[1] + half] = _decide_bit(
                    right_llr[0], rp, info_set
                )
            else:
                position = _right_down(position)
        elif _all_ready(left_bit):
            rv = np.array(
                [
                    g_operation(up_llr[i], up_llr[i + half], left_bit[i])
                    for i in range(half)
                ]
            )
            llr_matrix[position[0] + 1, position[1] + half : position[1] + span] = rv
        elif not _all_ready(left_llr):
            lv = np.array([f_operation(up_llr[i], up_llr[i + half]) for i in range(half)])
            llr_matrix[position[0] + 1, position[1] : position[1] + half] = lv
        else:
            if position[0] == position[2] - 1:
                lp = position[1]
                bit_matrix[position[0] + 1, position[1]] = _decide_bit(
                    left_llr[0], lp, info_set
                )
            else:
                position = _left_down(position)

    return llr_matrix, bit_matrix


def _path_metric(llr_leaf, u_bit):
    hard = 0 if llr_leaf >= 0 else 1
    return 0.0 if u_bit == hard else abs(llr_leaf)


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.info_set = set(int(i) for i in self.info_indices)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        llr_m, bit_m = _init_matrices(llr_ch, n, N)
        paths = [(llr_m, bit_m, 0.0)]
        split_positions = list(self.info_indices)

        split_loc = 0
        while split_loc < len(split_positions):
            split_pos = int(split_positions[split_loc])
            new_paths = []

            for llr_m, bit_m, pm in paths:
                llr_m, bit_m = _sc_step_to_split(
                    llr_m, bit_m, self.info_set, split_pos, n, N
                )
                leaf_llr = llr_m[n, split_pos]

                if split_pos not in self.info_set:
                    u_bit = 0
                    bit_m[n, split_pos] = u_bit
                    new_paths.append((llr_m, bit_m, pm + _path_metric(leaf_llr, u_bit)))
                else:
                    for u_bit in (0, 1):
                        lm = llr_m.copy()
                        bm = bit_m.copy()
                        bm[n, split_pos] = u_bit
                        new_paths.append(
                            (lm, bm, pm + _path_metric(leaf_llr, u_bit))
                        )

            new_paths.sort(key=lambda x: x[2])
            paths = new_paths[: self.list_size]
            split_loc += 1

        if self.crc_length > 0:
            valid = []
            for _, bit_m, pm in paths:
                u = np.nan_to_num(bit_m[n], nan=0).astype(np.int8)
                payload = u[self.info_indices]
                if crc_check(payload, self.crc_length):
                    valid.append((bit_m, pm))
            if valid:
                bit_m, pm = min(valid, key=lambda x: x[1])
            else:
                bit_m, pm = paths[0][1], paths[0][2]
        else:
            bit_m, pm = paths[0][1], paths[0][2]

        u_hat = np.nan_to_num(bit_m[n], nan=0).astype(np.int8)
        return u_hat, pm
