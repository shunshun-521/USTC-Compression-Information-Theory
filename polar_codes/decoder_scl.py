"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    _all_decided,
    _leftdown,
    _rightdown,
    _up,
    _get_up_bit,
    _get_left_llr,
    _get_right_llr,
)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = 0x07 if crc_length == 8 else 0x8005
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in info_bits:
        reg = ((reg << 1) | int(bit)) & mask
        if reg & (1 << (crc_length - 1)):
            reg ^= poly
    for _ in range(crc_length):
        reg = (reg << 1) & mask
        if reg & (1 << (crc_length - 1)):
            reg ^= poly
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


def _get_up_loc(bit_matrix):
    N = bit_matrix[0].size
    n = int(np.log2(N))
    detect_array = bit_matrix[n]
    detect = N - 1
    for i in range(N):
        if not np.isnan(detect_array[i]):
            continue
        detect = i - 1
        break
    if detect == -1:
        return [0, 0]
    if detect % 2 == 0:
        return [n - 1, detect]
    return [n - 1, detect - 1]


def _pm_update(llr_slice, bit_slice):
    pm = 0.0
    for llr, bit in zip(llr_slice, bit_slice):
        hard = 0 if llr >= 0 else 1
        if bit != hard:
            pm += abs(llr)
    return pm


def _sc_step(llr_matrix, bit_matrix, info_positions, frozen_bits, split_pos):
    N = int(bit_matrix[0].size)
    n = int(np.log2(N))
    loc = _get_up_loc(bit_matrix)
    position = [loc[0], loc[1], n, N]

    while np.isnan(bit_matrix[n][split_pos]):
        up_llr = llr_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])]
        up_bit = bit_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])]
        span = 2 ** (position[2] - position[0] - 1)
        left_llr = llr_matrix[position[0] + 1][position[1]:position[1] + span]
        left_bit = bit_matrix[position[0] + 1][position[1]:position[1] + span]
        right_llr = llr_matrix[position[0] + 1][position[1] + span:position[1] + 2 * span]
        right_bit = bit_matrix[position[0] + 1][position[1] + span:position[1] + 2 * span]

        if _all_decided(up_bit):
            position = _up(position)
        else:
            if _all_decided(right_bit):
                up_bit_new = _get_up_bit(left_bit, right_bit)
                bit_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])] = up_bit_new
            else:
                if _all_decided(right_llr):
                    if position[0] == position[2] - 1:
                        right_bit_pos = position[1] + 1
                        if right_bit_pos in info_positions:
                            rb = 0 if right_llr[0] >= 0 else 1
                        else:
                            rb = 0
                        bit_matrix[position[0] + 1][position[1] + span] = rb
                    else:
                        position = _rightdown(position)
                else:
                    if _all_decided(left_bit):
                        right_llr_new = _get_right_llr(left_bit, up_llr)
                        llr_matrix[position[0] + 1][position[1] + span:position[1] + 2 * span] = right_llr_new
                    else:
                        if not _all_decided(left_llr):
                            left_llr_new = _get_left_llr(up_llr)
                            llr_matrix[position[0] + 1][position[1]:position[1] + span] = left_llr_new
                        else:
                            if position[0] == position[2] - 1:
                                left_bit_pos = position[1]
                                if left_bit_pos in info_positions:
                                    lb = 0 if left_llr[0] >= 0 else 1
                                else:
                                    lb = 0
                                bit_matrix[position[0] + 1][position[1]] = lb
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
        self.info_positions = set(self.info_indices.tolist())

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1 and self.crc_length == 0:
            from decoder_sc import sc_decode
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        N, n = self.N, self.n

        llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
        bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
        llr_matrix[0] = llr_ch

        llr_list = [llr_matrix.copy()]
        bit_list = [bit_matrix.copy()]
        pm_list = [0.0]

        split_positions = list(self.info_indices)
        split_loc = 0

        while split_loc < len(split_positions):
            split_pos = split_positions[split_loc]
            prev_pos = split_positions[split_loc - 1] if split_loc > 0 else -1
            new_llr_list = []
            new_bit_list = []
            new_pm_list = []

            for llr_m, bit_m, pm in zip(llr_list, bit_list, pm_list):
                llr_m, bit_m = _sc_step(
                    llr_m.copy(), bit_m.copy(), self.info_positions, self.frozen_bits, split_pos
                )
                llr_slice = llr_m[n][prev_pos + 1:split_pos + 1]
                bit_slice = bit_m[n][prev_pos + 1:split_pos + 1]

                new_llr_list.append(llr_m)
                new_bit_list.append(bit_m.copy())
                new_pm_list.append(pm + _pm_update(llr_slice, bit_slice))

                bit_wrong = bit_m.copy()
                bit_wrong[n][split_pos] = 1 - bit_wrong[n][split_pos]
                wrong_slice = bit_wrong[n][prev_pos + 1:split_pos + 1]
                new_llr_list.append(llr_m.copy())
                new_bit_list.append(bit_wrong)
                new_pm_list.append(pm + _pm_update(llr_slice, wrong_slice))

            order = np.argsort(new_pm_list)
            keep = order[: self.list_size]
            llr_list = [new_llr_list[i] for i in keep]
            bit_list = [new_bit_list[i] for i in keep]
            pm_list = [new_pm_list[i] for i in keep]
            split_loc += 1

        if split_positions and split_positions[-1] != N - 1:
            final_pos = N - 1
            prev_pos = split_positions[-1]
            for i in range(len(llr_list)):
                llr_list[i], bit_list[i] = _sc_step(
                    llr_list[i], bit_list[i], self.info_positions, self.frozen_bits, final_pos
                )
                pm_list[i] += _pm_update(
                    llr_list[i][n][prev_pos + 1:N], bit_list[i][n][prev_pos + 1:N]
                )

        order = np.argsort(pm_list)
        best_u = None
        best_pm = pm_list[order[0]]

        if self.crc_length > 0:
            for idx in order:
                u_hat = bit_list[idx][n].astype(int)
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    return u_hat, pm_list[idx]

        best_u = bit_list[order[0]][n].astype(int)
        return best_u, best_pm
