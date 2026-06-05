"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _all_num,
    _get_left_bit,
    _get_left_llr,
    _get_right_bit,
    _get_right_llr,
    _get_up_bit,
    _leftdown,
    _rightdown,
    _up,
    sc_decode,
)


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


def _get_up_loc(bit_matrix):
    n = int(math.log2(bit_matrix.shape[1]))
    for i in range(n + 1):
        for j in range(bit_matrix.shape[1]):
            if np.isnan(bit_matrix[i, j]):
                return [i, j]
    return [n, 0]


def _path_metric(llr, bit):
    hard = 0 if llr >= 0 else 1
    return 0.0 if bit == hard else abs(llr)


def _sc_stepping(llr_matrix, bit_matrix, info_set, frozen_val, split_pos):
    """SC 推进至 split_pos 判决完成。"""
    N = bit_matrix.shape[1]
    n = int(math.log2(N))
    loc = _get_up_loc(bit_matrix)
    position = [loc[0], loc[1], n, N]

    while bit_matrix[n, split_pos] not in (0, 1):
        up_llr = llr_matrix[position[0]][
            position[1]:position[1] + 2 ** (position[2] - position[0])
        ]
        up_bit = bit_matrix[position[0]][
            position[1]:position[1] + 2 ** (position[2] - position[0])
        ]
        left_llr = llr_matrix[position[0] + 1][
            position[1]:position[1] + 2 ** (position[2] - position[0] - 1)
        ]
        left_bit = bit_matrix[position[0] + 1][
            position[1]:position[1] + 2 ** (position[2] - position[0] - 1)
        ]
        right_llr = llr_matrix[position[0] + 1][
            position[1] + 2 ** (position[2] - position[0] - 1):
            position[1] + 2 ** (position[2] - position[0])
        ]
        right_bit = bit_matrix[position[0] + 1][
            position[1] + 2 ** (position[2] - position[0] - 1):
            position[1] + 2 ** (position[2] - position[0])
        ]

        if _all_num(up_bit) == 1:
            position = _up(position)
        else:
            if _all_num(right_bit) == 1:
                up_bit_val = _get_up_bit(left_bit, right_bit)
                bit_matrix[position[0]][
                    position[1]:position[1] + 2 ** (position[2] - position[0])
                ] = up_bit_val.copy()
            else:
                if _all_num(right_llr) == 1:
                    if position[0] == position[2] - 1:
                        right_bit_pos = position[1] + 1
                        bit_matrix[position[0] + 1][
                            position[1] + 2 ** (position[2] - position[0] - 1):
                            position[1] + 2 ** (position[2] - position[0])
                        ] = _get_right_bit(
                            right_llr, info_set, frozen_val, right_bit_pos
                        )
                    else:
                        position = _rightdown(position)
                else:
                    if _all_num(left_bit) == 1:
                        llr_matrix[position[0] + 1][
                            position[1] + 2 ** (position[2] - position[0] - 1):
                            position[1] + 2 ** (position[2] - position[0])
                        ] = _get_right_llr(left_bit, up_llr)
                    else:
                        if _all_num(left_llr) == 0:
                            llr_matrix[position[0] + 1][
                                position[1]:position[1] + 2 ** (position[2] - position[0] - 1)
                            ] = _get_left_llr(up_llr)
                        else:
                            if position[0] == position[2] - 1:
                                left_bit_pos = position[1]
                                bit_matrix[position[0] + 1][
                                    position[1]:position[1] + 2 ** (position[2] - position[0] - 1)
                                ] = _get_left_bit(
                                    left_llr, info_set, frozen_val, left_bit_pos
                                )
                            else:
                                position = _leftdown(position)

    return llr_matrix, bit_matrix


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.info_set = set(self.info_indices.tolist())
        self.frozen_val = 0

    def decode(self, llr_ch):
        if self.list_size == 1:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        y_llr = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        llr_matrix = np.ones((n + 1, N), dtype=np.float64)
        llr_matrix[llr_matrix == 1] = np.nan
        bit_matrix = llr_matrix.copy()
        llr_matrix[0] = y_llr

        paths = [(llr_matrix.copy(), bit_matrix.copy(), 0.0)]
        info_positions = sorted(self.info_indices.tolist())

        for phi in info_positions:
            new_paths = []
            for llr_m, bit_m, pm in paths:
                llr_m, bit_m = _sc_stepping(
                    llr_m, bit_m, self.info_set, self.frozen_val, phi
                )
                llr_leaf = llr_m[n, phi] if phi > 0 else llr_m[0, 0]
                # 使用已判决位的 LLR 近似路径度量
                root_llr = llr_m[0, 0]

                for bit in (0, 1):
                    bm = bit_m.copy()
                    bm[n, phi] = bit
                    new_pm = pm + _path_metric(root_llr, bit)
                    new_paths.append((llr_m.copy(), bm, new_pm))

            new_paths.sort(key=lambda x: x[2])
            paths = new_paths[: self.list_size]

        # 完成剩余冻结位
        last_phi = info_positions[-1] if info_positions else -1
        final_paths = []
        for llr_m, bit_m, pm in paths:
            llr_m, bit_m = _sc_stepping(
                llr_m, bit_m, self.info_set, self.frozen_val, N - 1
            )
            final_paths.append((llr_m, bit_m, pm))
        paths = final_paths

        if self.crc_length > 0:
            valid = []
            for _, bit_m, pm in paths:
                u = bit_m[n].astype(int)
                if crc_check(u[self.info_indices], self.crc_length):
                    valid.append((_, bit_m, pm))
            if valid:
                paths = valid

        best = min(paths, key=lambda x: x[2])
        return best[1][n].astype(int), best[2]
