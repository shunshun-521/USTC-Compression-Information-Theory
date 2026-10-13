"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

import polar_sc_ref as ref
from decoder_sc import sc_decode

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    mask = (1 << crc_length) - 1
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    mask = (1 << crc_length) - 1
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg == 0


def _sc_step_to_bit(llr_matrix, bit_matrix, information_pos, frozen_bit, bit_pos):
    """运行 SC 直至完成 bit_pos 判决。"""
    N = int(bit_matrix[0].size)
    n = int(np.log2(N))
    loc = ref.get_up_loc(bit_matrix)
    position = [loc[0], loc[1], n, N]

    while bit_matrix[n][bit_pos] != 0 and bit_matrix[n][bit_pos] != 1:
        span = 2 ** (position[2] - position[0])
        up_llr = llr_matrix[position[0]][position[1] : position[1] + span]
        up_bit = bit_matrix[position[0]][position[1] : position[1] + span]
        half = span // 2
        left_llr = llr_matrix[position[0] + 1][position[1] : position[1] + half]
        left_bit = bit_matrix[position[0] + 1][position[1] : position[1] + half]
        right_llr = llr_matrix[position[0] + 1][position[1] + half : position[1] + span]
        right_bit = bit_matrix[position[0] + 1][position[1] + half : position[1] + span]

        if ref.all_num(up_bit) == 1:
            position = ref.up(position)
        else:
            if ref.all_num(right_bit) == 1:
                up_bit = ref.get_up_bit(left_bit, right_bit)
                bit_matrix[position[0]][position[1] : position[1] + span] = up_bit.copy()
            else:
                if ref.all_num(right_llr) == 1:
                    if position[0] == position[2] - 1:
                        right_bit_pos = position[1] + 1
                        right_bit_val = ref.get_right_bit(
                            right_llr, information_pos, frozen_bit, right_bit_pos
                        )
                        bit_matrix[position[0] + 1][position[1] + half : position[1] + span] = (
                            right_bit_val
                        )
                    else:
                        position = ref.rightdown(position)
                else:
                    if ref.all_num(left_bit) == 1:
                        right_llr = ref.get_right_llr(left_bit, up_llr)
                        llr_matrix[position[0] + 1][position[1] + half : position[1] + span] = (
                            right_llr
                        )
                    else:
                        if ref.all_num(left_llr) == 0:
                            left_llr = ref.get_left_llr(up_llr)
                            llr_matrix[position[0] + 1][position[1] : position[1] + half] = (
                                left_llr
                            )
                        else:
                            if position[0] == position[2] - 1:
                                left_bit_pos = position[1]
                                left_bit_val = ref.get_left_bit(
                                    left_llr, information_pos, frozen_bit, left_bit_pos
                                )
                                bit_matrix[position[0] + 1][position[1] : position[1] + half] = (
                                    left_bit_val
                                )
                            else:
                                position = ref.leftdown(position)
    return llr_matrix, bit_matrix


class SCLDecoder:
    """SCL 译码器（矩阵状态 + 路径复制）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_pos = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1:
            u = sc_decode(llr_ch, self.frozen_bits)
            return u, 0.0

        N, n = self.N, self.n
        llr0 = np.ones((n + 1, N))
        llr0[llr0 == 1] = np.nan
        bit0 = llr0.copy()
        llr0[0] = llr_ch

        paths = [(0.0, llr0.copy(), bit0.copy())]

        for bit_pos in range(N):
            new_paths = []
            for pm, llr_m, bit_m in paths:
                llr_m, bit_m = _sc_step_to_bit(
                    llr_m, bit_m, self.info_pos, 0, bit_pos
                )
                llr_leaf = llr_m[n][bit_pos]
                if self.frozen_bits[bit_pos]:
                    hard = 0 if llr_leaf >= 0 else 1
                    penalty = 0.0 if hard == 0 else abs(llr_leaf)
                    bit_m[n][bit_pos] = 0
                    new_paths.append((pm + penalty, llr_m, bit_m))
                else:
                    for u in (0, 1):
                        hard = 0 if llr_leaf >= 0 else 1
                        penalty = 0.0 if u == hard else abs(llr_leaf)
                        bm = bit_m.copy()
                        bm[n][bit_pos] = u
                        new_paths.append((pm + penalty, llr_m.copy(), bm))

            new_paths.sort(key=lambda x: x[0])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for pm, _l, bit_m in paths:
                u = bit_m[n].astype(np.int8)
                if crc_check(u[self.info_pos], self.crc_length):
                    valid.append((pm, _l, bit_m))
            if valid:
                paths = valid

        best = min(paths, key=lambda x: x[0])
        return best[2][n].astype(np.int8), best[0]
