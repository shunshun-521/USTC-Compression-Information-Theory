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
    g_operation,
)


CRC_POLYNOMIALS = {8: 0x07, 16: 0x8005}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC_POLYNOMIALS[crc_length]
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
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


def _get_up_loc(bit_matrix):
    N = int(bit_matrix[0].size)
    n = int(np.log2(N))
    detect_array = bit_matrix[n]
    detect = N - 1
    for i in range(N):
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


def _get_pm_update(llr_array, bit_array):
    pm = 0.0
    for i in range(len(llr_array)):
        expected = 1 - 2 * bit_array[i]
        if np.sign(llr_array[i]) != np.sign(expected):
            pm += abs(llr_array[i])
    return pm


def _sc_stepping_decoder(llr_matrix, bit_matrix, frozen_bits, split_pos):
    N = int(bit_matrix[0].size)
    n = int(np.log2(N))
    loc = _get_up_loc(bit_matrix)
    position = [loc[0], loc[1], n, N]

    while bit_matrix[n][split_pos] != 0 and bit_matrix[n][split_pos] != 1:
        row, col, depth, _ = position
        span = 2 ** (depth - row)
        half = 2 ** (depth - row - 1)

        up_llr = llr_matrix[row, col:col + span]
        left_bit = bit_matrix[row + 1, col:col + half]
        right_bit = bit_matrix[row + 1, col + half:col + span]
        left_llr = llr_matrix[row + 1, col:col + half]
        right_llr = llr_matrix[row + 1, col + half:col + span]
        up_bit = bit_matrix[row, col:col + span]

        if _all_num(up_bit):
            position = _up(position)
        elif _all_num(right_bit):
            up_bit_new = _get_up_bit(left_bit, right_bit)
            bit_matrix[row, col:col + span] = up_bit_new.copy()
        elif _all_num(right_llr):
            if row == depth - 1:
                right_bit_pos = col + 1
                right_bit_val = _get_right_bit(
                    right_llr[0], frozen_bits, right_bit_pos
                )
                bit_matrix[row + 1, col + half:col + span] = right_bit_val
            else:
                position = _rightdown(position)
        elif _all_num(left_bit):
            right_llr_new = _get_right_llr(left_bit, up_llr)
            llr_matrix[row + 1, col + half:col + span] = right_llr_new
        elif not _all_num(left_llr):
            left_llr_new = _get_left_llr(up_llr)
            llr_matrix[row + 1, col:col + half] = left_llr_new
        else:
            if row == depth - 1:
                left_bit_pos = col
                left_bit_val = _get_left_bit(
                    left_llr[0], frozen_bits, left_bit_pos
                )
                bit_matrix[row + 1, col:col + half] = left_bit_val
            else:
                position = _leftdown(position)

    return llr_matrix, bit_matrix


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        y_llr = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        split_pos = list(self.info_indices)

        llr_matrix = np.ones((n + 1, N), dtype=np.float64)
        llr_matrix[llr_matrix == 1] = np.nan
        bit_matrix = llr_matrix.copy()
        llr_matrix[0] = y_llr

        llr_list = [llr_matrix.copy()]
        bit_list = [bit_matrix.copy()]
        pm_list = [0.0]
        split_loc = 0
        l_now = 1

        while split_loc < len(split_pos):
            prev_pos = split_pos[split_loc - 1] if split_loc > 0 else -1
            cur_pos = split_pos[split_loc]

            new_llr_list = []
            new_bit_list = []
            new_pm_list = []

            for i in range(l_now):
                llr_temp = llr_list[i].copy()
                bit_temp = bit_list[i].copy()
                pm_temp = pm_list[i]

                llr_out, bit_out = _sc_stepping_decoder(
                    llr_temp, bit_temp, self.frozen_bits, cur_pos
                )

                llr_slice = llr_out[n, prev_pos + 1:cur_pos + 1]
                bit_slice = bit_out[n, prev_pos + 1:cur_pos + 1]
                pm_update = _get_pm_update(llr_slice, bit_slice)

                new_llr_list.append(llr_out)
                new_bit_list.append(bit_out)
                new_pm_list.append(pm_temp + pm_update)

                if not self.frozen_bits[cur_pos]:
                    bit_wrong = bit_out.copy()
                    bit_wrong[n, cur_pos] = 1 - bit_wrong[n, cur_pos]
                    bit_slice_w = bit_wrong[n, prev_pos + 1:cur_pos + 1]
                    pm_wrong = pm_temp + _get_pm_update(llr_slice, bit_slice_w)
                    new_llr_list.append(llr_out.copy())
                    new_bit_list.append(bit_wrong)
                    new_pm_list.append(pm_wrong)

            order = np.argsort(new_pm_list)[:self.list_size]
            llr_list = [new_llr_list[i] for i in order]
            bit_list = [new_bit_list[i] for i in order]
            pm_list = [new_pm_list[i] for i in order]
            l_now = len(pm_list)
            split_loc += 1

        if split_pos and split_pos[-1] != N - 1:
            for i in range(l_now):
                llr_temp = llr_list[i].copy()
                bit_temp = bit_list[i].copy()
                pm_temp = pm_list[i]
                llr_out, bit_out = _sc_stepping_decoder(
                    llr_temp, bit_temp, self.frozen_bits, N - 1
                )
                prev_pos = split_pos[-1]
                pm_update = _get_pm_update(
                    llr_out[n, prev_pos + 1:N],
                    bit_out[n, prev_pos + 1:N],
                )
                llr_list[i] = llr_out
                bit_list[i] = bit_out
                pm_list[i] = pm_temp + pm_update

        order = np.argsort(pm_list)
        best_u = None
        best_pm = pm_list[order[0]]

        for idx in order:
            u_hat = bit_list[idx][n].astype(int)
            if self.crc_length > 0:
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    return u_hat, pm_list[idx]
            if best_u is None:
                best_u = u_hat
                best_pm = pm_list[idx]

        return best_u, best_pm
