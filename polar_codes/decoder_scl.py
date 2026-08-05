"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import _sc_decode_core, _frozen_to_info_pos
from polar_common import all_num, get_pm_update, get_up_loc
from polar_common import (
    get_left_bit,
    get_left_llr,
    get_right_bit,
    get_right_llr,
    get_up_bit,
    leftdown,
    rightdown,
    up,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg = (reg << 1) | (int(bit) & 1)
        if reg & (1 << crc_length):
            reg ^= poly
    mask = (1 << crc_length) - 1
    for _ in range(crc_length):
        reg <<= 1
        if reg & (1 << crc_length):
            reg ^= poly
    return reg & mask


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


def _sc_stepping_decoder(llr_matrix, bit_matrix, information_pos, frozen_bit, split_pos):
    """SC 译码到指定比特位置。"""
    N = int(bit_matrix[0].size)
    n = int(np.log2(N))
    loc = get_up_loc(bit_matrix)
    position = [loc[0], loc[1], n, N]

    while bit_matrix[n][split_pos] != 0 and bit_matrix[n][split_pos] != 1:
        up_llr = llr_matrix[position[0]][
            position[1] : position[1] + 2 ** (position[2] - position[0])
        ]
        left_bit = bit_matrix[position[0] + 1][
            position[1] : position[1] + 2 ** (position[2] - position[0] - 1)
        ]
        right_llr = llr_matrix[position[0] + 1][
            position[1]
            + 2 ** (position[2] - position[0] - 1) : position[1]
            + 2 ** (position[2] - position[0])
        ]
        right_bit = bit_matrix[position[0] + 1][
            position[1]
            + 2 ** (position[2] - position[0] - 1) : position[1]
            + 2 ** (position[2] - position[0])
        ]
        left_llr = llr_matrix[position[0] + 1][
            position[1] : position[1] + 2 ** (position[2] - position[0] - 1)
        ]

        if all_num(bit_matrix[position[0]][position[1] : position[1] + 2 ** (position[2] - position[0])]) == 1:
            position = up(position)
        else:
            if all_num(right_bit) == 1:
                up_bit = get_up_bit(left_bit, right_bit)
                bit_matrix[position[0]][
                    position[1] : position[1] + 2 ** (position[2] - position[0])
                ] = up_bit.copy()
            else:
                if all_num(right_llr) == 1:
                    if position[0] == position[2] - 1:
                        right_bit_pos = position[1] + 1
                        right_bit_val = get_right_bit(
                            right_llr[0], information_pos, frozen_bit, right_bit_pos
                        )
                        bit_matrix[position[0] + 1][
                            position[1]
                            + 2 ** (position[2] - position[0] - 1) : position[1]
                            + 2 ** (position[2] - position[0])
                        ] = right_bit_val
                    else:
                        position = rightdown(position)
                else:
                    if all_num(left_bit) == 1:
                        right_llr_new = get_right_llr(left_bit, up_llr)
                        llr_matrix[position[0] + 1][
                            position[1]
                            + 2 ** (position[2] - position[0] - 1) : position[1]
                            + 2 ** (position[2] - position[0])
                        ] = right_llr_new
                    else:
                        if all_num(left_llr) == 0:
                            left_llr_new = get_left_llr(up_llr)
                            llr_matrix[position[0] + 1][
                                position[1] : position[1]
                                + 2 ** (position[2] - position[0] - 1)
                            ] = left_llr_new
                        else:
                            if position[0] == position[2] - 1:
                                left_bit_pos = position[1]
                                left_bit_val = get_left_bit(
                                    left_llr[0], information_pos, frozen_bit, left_bit_pos
                                )
                                bit_matrix[position[0] + 1][
                                    position[1] : position[1]
                                    + 2 ** (position[2] - position[0] - 1)
                                ] = left_bit_val
                            else:
                                position = leftdown(position)
    return llr_matrix, bit_matrix


def _scl_decode(y_llr, information_pos, frozen_bit, list_size, crc_length=0):
    """SCL 译码核心。"""
    N = y_llr.size
    n = int(np.log2(N))
    llr_matrix = np.ones((n + 1, N))
    llr_matrix[llr_matrix == 1] = np.nan
    bit_matrix = llr_matrix.copy()
    llr_matrix[0] = y_llr

    split_pos = list(information_pos)
    llr_list = [llr_matrix]
    bit_list = [bit_matrix]
    pm_list = [0.0]
    split_loc = 0
    l_now = 1

    while split_loc < len(split_pos):
        prev = split_pos[split_loc - 1] if split_loc > 0 else -1
        cur = split_pos[split_loc]
        new_llr_list = []
        new_bit_list = []
        new_pm_list = []

        for i in range(l_now):
            llr_temp = llr_list[i].copy()
            bit_temp = bit_list[i].copy()
            pm_temp = pm_list[i]
            llr_out, bit_out = _sc_stepping_decoder(
                llr_temp, bit_temp, information_pos, frozen_bit, cur
            )
            seg_llr = llr_out[n][prev + 1 : cur + 1]
            seg_bit = bit_out[n][prev + 1 : cur + 1]
            pm_ok = pm_temp + get_pm_update(seg_llr, seg_bit, "hf")
            new_llr_list.append(llr_out)
            new_bit_list.append(bit_out)
            new_pm_list.append(pm_ok)

            bit_wrong = bit_out.copy()
            bit_wrong[n][cur] = 1 - bit_wrong[n][cur]
            seg_bit_w = bit_wrong[n][prev + 1 : cur + 1]
            pm_bad = pm_temp + get_pm_update(seg_llr, seg_bit_w, "hf")
            new_llr_list.append(llr_out.copy())
            new_bit_list.append(bit_wrong)
            new_pm_list.append(pm_bad)

        order = np.argsort(new_pm_list)[:list_size]
        llr_list = [new_llr_list[i] for i in order]
        bit_list = [new_bit_list[i] for i in order]
        pm_list = [new_pm_list[i] for i in order]
        l_now = len(pm_list)
        split_loc += 1

    if split_pos and split_pos[-1] != N - 1:
        prev = split_pos[-1]
        for i in range(l_now):
            llr_out, bit_out = _sc_stepping_decoder(
                llr_list[i].copy(), bit_list[i].copy(), information_pos, frozen_bit, N - 1
            )
            seg_llr = llr_out[n][prev + 1 : N]
            seg_bit = bit_out[n][prev + 1 : N]
            pm_list[i] += get_pm_update(seg_llr, seg_bit, "hf")
            llr_list[i] = llr_out
            bit_list[i] = bit_out

    order = np.argsort(pm_list)
    best_u = None
    best_pm = None
    for idx in order:
        u_cand = bit_list[idx][n].astype(int)
        if crc_length > 0:
            info_bits = u_cand[information_pos]
            if crc_check(info_bits, crc_length):
                return u_cand, pm_list[idx]
        elif best_u is None:
            best_u = u_cand
            best_pm = pm_list[idx]
    if best_u is None:
        best_u = bit_list[order[0]][n].astype(int)
        best_pm = pm_list[order[0]]
    return best_u, best_pm


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.information_pos = _frozen_to_info_pos(frozen_bits)

    def decode(self, llr_ch):
        u_hat, pm = _scl_decode(
            np.asarray(llr_ch, dtype=np.float64),
            self.information_pos,
            0,
            self.list_size,
            self.crc_length,
        )
        return u_hat, pm
