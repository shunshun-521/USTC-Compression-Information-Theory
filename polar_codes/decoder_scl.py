"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
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


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    crc_bits = np.array(
        [(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    return np.array_equal(bits[-crc_length:], crc_encode(bits[:-crc_length], crc_length)[-crc_length:])


def _get_up_loc(bit_matrix):
    N = int(bit_matrix[0].size)
    n = int(np.log2(N))
    detect_array = bit_matrix[n]
    detect = -1
    for i in range(N):
        if detect_array[i] not in (0, 1):
            detect = i - 1
            break
    if detect % 2 == 0:
        loc_row = n - 1
        loc_col = detect
    else:
        loc_row = n - 1
        loc_col = detect - 1
    if detect == -1:
        loc_row = 0
        loc_col = 0
    return [loc_row, loc_col]


def _pm_update(llr_array, bit_array):
    pm = 0.0
    for llr, bit in zip(llr_array, bit_array):
        hard = 0 if llr >= 0 else 1
        if bit != hard:
            pm += abs(llr)
    return pm


def _sc_stepping_decoder(llr_matrix, bit_matrix, information_pos, frozen_bit, split_pos):
    N = int(bit_matrix.shape[1])
    n = int(np.log2(N))
    loc = _get_up_loc(bit_matrix)
    position = [loc[0], loc[1], n, N]

    while bit_matrix[n][split_pos] not in (0, 1):
        up_llr = llr_matrix[position[0]][
            position[1] : position[1] + 2 ** (position[2] - position[0])
        ]
        up_bit = bit_matrix[position[0]][
            position[1] : position[1] + 2 ** (position[2] - position[0])
        ]
        left_llr = llr_matrix[position[0] + 1][
            position[1] : position[1] + 2 ** (position[2] - position[0] - 1)
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

        if _all_num(up_bit) == 1:
            position = _up(position)
        else:
            if _all_num(right_bit) == 1:
                up_bit_new = _get_up_bit(left_bit, right_bit)
                bit_matrix[position[0]][
                    position[1] : position[1] + 2 ** (position[2] - position[0])
                ] = up_bit_new.copy()
            else:
                if _all_num(right_llr) == 1:
                    if position[0] == position[2] - 1:
                        right_bit_pos = position[1] + 1
                        right_bit_val = _get_right_bit(
                            right_llr[0], information_pos, frozen_bit, right_bit_pos
                        )
                        bit_matrix[position[0] + 1][
                            position[1]
                            + 2 ** (position[2] - position[0] - 1) : position[1]
                            + 2 ** (position[2] - position[0])
                        ] = right_bit_val
                    else:
                        position = _rightdown(position)
                else:
                    if _all_num(left_bit) == 1:
                        right_llr_new = _get_right_llr(left_bit, up_llr)
                        llr_matrix[position[0] + 1][
                            position[1]
                            + 2 ** (position[2] - position[0] - 1) : position[1]
                            + 2 ** (position[2] - position[0])
                        ] = right_llr_new
                    else:
                        if _all_num(left_llr) == 0:
                            left_llr_new = _get_left_llr(up_llr)
                            llr_matrix[position[0] + 1][
                                position[1] : position[1]
                                + 2 ** (position[2] - position[0] - 1)
                            ] = left_llr_new
                        else:
                            if position[0] == position[2] - 1:
                                left_bit_pos = position[1]
                                left_bit_val = _get_left_bit(
                                    left_llr[0],
                                    information_pos,
                                    frozen_bit,
                                    left_bit_pos,
                                )
                                bit_matrix[position[0] + 1][
                                    position[1] : position[1]
                                    + 2 ** (position[2] - position[0] - 1)
                                ] = left_bit_val
                            else:
                                position = _leftdown(position)

    return llr_matrix, bit_matrix


def _scl_decode_core(y_llr, information_pos, list_size, crc_length=0):
    N = y_llr.size
    n = int(np.log2(N))
    llr_matrix = np.ones((n + 1, N), dtype=np.float64)
    llr_matrix[llr_matrix == 1] = np.nan
    bit_matrix = llr_matrix.copy()
    llr_matrix[0] = y_llr

    split_pos = np.asarray(information_pos, dtype=int)
    llr_list = [llr_matrix.copy()]
    bit_list = [bit_matrix.copy()]
    pm_list = [0.0]
    split_loc = 0
    split_len = len(split_pos)
    l_now = 1

    while split_len - 1 >= split_loc:
        new_llr_list = []
        new_bit_list = []
        new_pm_list = []
        for i in range(l_now):
            llr_temp = llr_list[i].copy()
            bit_temp = bit_list[i].copy()
            pm_temp = pm_list[i]
            llr_out, bit_out = _sc_stepping_decoder(
                llr_temp, bit_temp, information_pos, 0, split_pos[split_loc]
            )
            prev_end = split_pos[split_loc - 1] + 1 if split_loc > 0 else 0
            cur_end = split_pos[split_loc] + 1
            pm_add = _pm_update(
                llr_out[n][prev_end:cur_end], bit_out[n][prev_end:cur_end]
            )
            new_llr_list.append(llr_out)
            new_bit_list.append(bit_out)
            new_pm_list.append(pm_temp + pm_add)
            bit_wrong = bit_out.copy()
            bit_wrong[n][split_pos[split_loc]] = 1 - bit_wrong[n][split_pos[split_loc]]
            pm_wrong = pm_temp + _pm_update(
                llr_out[n][prev_end:cur_end], bit_wrong[n][prev_end:cur_end]
            )
            new_llr_list.append(llr_out.copy())
            new_bit_list.append(bit_wrong)
            new_pm_list.append(pm_wrong)

        order = np.argsort(new_pm_list)[:list_size]
        llr_list = [new_llr_list[i] for i in order]
        bit_list = [new_bit_list[i] for i in order]
        pm_list = [new_pm_list[i] for i in order]
        l_now = len(pm_list)
        split_loc += 1

    if split_pos[-1] != N - 1:
        for i in range(l_now):
            llr_temp = llr_list[i].copy()
            bit_temp = bit_list[i].copy()
            pm_temp = pm_list[i]
            llr_out, bit_out = _sc_stepping_decoder(
                llr_temp, bit_temp, information_pos, 0, N - 1
            )
            prev_end = split_pos[split_loc - 1] + 1
            pm_add = _pm_update(llr_out[n][prev_end:N], bit_out[n][prev_end:N])
            llr_list[i] = llr_out
            bit_list[i] = bit_out
            pm_list[i] = pm_temp + pm_add

    order = np.argsort(pm_list)
    best_u = bit_list[order[0]][n].astype(int)
    best_pm = pm_list[order[0]]

    if crc_length > 0:
        for idx in order:
            u_cand = bit_list[idx][n].astype(int)
            u_info = u_cand[information_pos]
            if crc_check(u_info, crc_length):
                return u_cand, pm_list[idx]
    return best_u, best_pm


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        if info_indices is None:
            self.info_indices = np.where(~self.frozen_bits)[0]
        else:
            self.info_indices = np.asarray(info_indices, dtype=np.int64)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size <= 1:
            return sc_decode(llr_ch, self.frozen_bits), 0.0
        return _scl_decode_core(
            llr_ch, self.info_indices, self.list_size, self.crc_length
        )
