"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    _all_filled,
    _get_left_bit,
    _get_left_llr,
    _get_right_bit,
    _get_right_llr,
    _get_up_bit,
    _leftdown,
    _rightdown,
    _up_position,
    sc_decode,
)


CRC_POLYNOMIALS = {8: 0x07, 16: 0x8005}


def _crc_polynomial(crc_length):
    if crc_length == 8:
        loc = [8, 2, 1, 0]
    elif crc_length == 16:
        loc = [16, 15, 2, 0]
    else:
        raise ValueError("crc_length must be 8 or 16")
    poly = [0] * (crc_length + 1)
    for i in loc:
        poly[i] = 1
    return poly[::-1]


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后（多项式长除法）。"""
    info_bits = [int(b) for b in np.asarray(info_bits, dtype=np.int8).ravel()]
    p = _crc_polynomial(crc_length)
    work = info_bits.copy()
    times = len(work)
    work.extend([0] * crc_length)
    for i in range(times):
        if work[i] == 1:
            for j in range(crc_length + 1):
                work[j + i] ^= p[j]
    check_code = work[-crc_length:]
    return np.array(info_bits + check_code, dtype=np.int8)


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确。"""
    bits = [int(b) for b in np.asarray(bits, dtype=np.int8).ravel()]
    info_len = len(bits) - crc_length
    if info_len <= 0:
        return False
    encoded = crc_encode(bits[:info_len], crc_length)
    return encoded.tolist() == bits


def _get_up_loc(bit_matrix):
    n = int(np.log2(bit_matrix.shape[1]))
    detect_array = bit_matrix[n]
    detect = 0
    for i in range(len(detect_array)):
        if detect_array[i] == 1 or detect_array[i] == 0:
            pass
        else:
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
    for i in range(len(llr_array)):
        hard = 0 if llr_array[i] >= 0 else 1
        if hard != bit_array[i]:
            pm += abs(llr_array[i])
    return pm


def _sc_stepping_decoder(llr_matrix, bit_matrix, information_pos, frozen_bit, split_pos):
    """运行 SC 至完成 split_pos 比特判决。"""
    N = int(bit_matrix.shape[1])
    n = int(math.log2(N))
    loc = _get_up_loc(bit_matrix)
    position = [loc[0], loc[1], n, N]

    while bit_matrix[n][split_pos] != 0 and bit_matrix[n][split_pos] != 1:
        span = 1 << (position[2] - position[0])
        up_llr = llr_matrix[position[0]][position[1]:position[1] + span]
        up_bit = bit_matrix[position[0]][position[1]:position[1] + span]
        left_llr = llr_matrix[position[0] + 1][position[1]:position[1] + span // 2]
        left_bit = bit_matrix[position[0] + 1][position[1]:position[1] + span // 2]
        right_llr = llr_matrix[position[0] + 1][
            position[1] + span // 2:position[1] + span
        ]
        right_bit = bit_matrix[position[0] + 1][
            position[1] + span // 2:position[1] + span
        ]

        if _all_filled(up_bit):
            position = _up_position(position)
        elif _all_filled(right_bit):
            up_bit_new = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0]][position[1]:position[1] + span] = up_bit_new.copy()
        elif _all_filled(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                right_bit_val = _get_right_bit(
                    right_llr, information_pos, frozen_bit, right_bit_pos
                )
                bit_matrix[position[0] + 1][
                    position[1] + span // 2:position[1] + span
                ] = right_bit_val
            else:
                position = _rightdown(position)
        elif _all_filled(left_bit):
            right_llr_new = _get_right_llr(left_bit, up_llr)
            llr_matrix[position[0] + 1][position[1] + span // 2:position[1] + span] = (
                right_llr_new
            )
        elif not _all_filled(left_llr):
            left_llr_new = _get_left_llr(up_llr)
            llr_matrix[position[0] + 1][position[1]:position[1] + span // 2] = left_llr_new
        else:
            if position[0] == position[2] - 1:
                left_bit_pos = position[1]
                left_bit_val = _get_left_bit(
                    left_llr, information_pos, frozen_bit, left_bit_pos
                )
                bit_matrix[position[0] + 1][position[1]:position[1] + span // 2] = (
                    left_bit_val
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
        self.information_pos = list(np.where(~self.frozen_bits)[0])
        self.info_indices = self.information_pos

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        N = self.N
        n = self.n
        frozen_bit = 0
        split_pos = [i for i in self.information_pos]

        llr_matrix = np.ones((n + 1, N), dtype=np.float64)
        llr_matrix[llr_matrix == 1] = np.nan
        bit_matrix = llr_matrix.copy()
        llr_matrix[0] = llr_ch

        llr_list = [llr_matrix.copy()]
        bit_list = [bit_matrix.copy()]
        pm_list = [0.0]
        split_loc = 0
        l_now = 1

        while split_loc < len(split_pos):
            sp = split_pos[split_loc]
            prev_sp = split_pos[split_loc - 1] if split_loc > 0 else -1
            new_llr_list = []
            new_bit_list = []
            new_pm_list = []

            for idx in range(l_now):
                llr_temp = llr_list[idx].copy()
                bit_temp = bit_list[idx].copy()
                pm_temp = pm_list[idx]
                llr_temp, bit_temp = _sc_stepping_decoder(
                    llr_temp, bit_temp, self.information_pos, frozen_bit, sp
                )
                llr_slice = llr_temp[n][prev_sp + 1:sp + 1]
                bit_slice = bit_temp[n][prev_sp + 1:sp + 1]

                new_llr_list.append(llr_temp)
                new_bit_list.append(bit_temp)
                new_pm_list.append(pm_temp + _pm_update(llr_slice, bit_slice))

                bit_wrong = bit_temp.copy()
                bit_wrong[n][sp] = 1 - bit_wrong[n][sp]
                wrong_slice = bit_wrong[n][prev_sp + 1:sp + 1]
                new_llr_list.append(llr_temp.copy())
                new_bit_list.append(bit_wrong)
                new_pm_list.append(pm_temp + _pm_update(llr_slice, wrong_slice))

            order = np.argsort(new_pm_list)
            keep = order[: self.list_size]
            llr_list = [new_llr_list[i] for i in keep]
            bit_list = [new_bit_list[i] for i in keep]
            pm_list = [new_pm_list[i] for i in keep]
            l_now = len(pm_list)
            split_loc += 1

        if split_pos and split_pos[-1] != N - 1:
            new_pm_list = []
            for idx in range(l_now):
                llr_temp = llr_list[idx].copy()
                bit_temp = bit_list[idx].copy()
                pm_temp = pm_list[idx]
                llr_temp, bit_temp = _sc_stepping_decoder(
                    llr_temp, bit_temp, self.information_pos, frozen_bit, N - 1
                )
                prev_sp = split_pos[-1]
                llr_slice = llr_temp[n][prev_sp + 1:N]
                bit_slice = bit_temp[n][prev_sp + 1:N]
                llr_list[idx] = llr_temp
                bit_list[idx] = bit_temp
                new_pm_list.append(pm_temp + _pm_update(llr_slice, bit_slice))
            pm_list = new_pm_list

        order = np.argsort(pm_list)
        if self.crc_length > 0:
            for idx in order:
                u_candidate = bit_list[idx][n].astype(np.int8)
                info_bits = u_candidate[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    return u_candidate, pm_list[idx]
            best = order[0]
            return bit_list[best][n].astype(np.int8), pm_list[best]

        best = order[0]
        return bit_list[best][n].astype(np.int8), pm_list[best]
