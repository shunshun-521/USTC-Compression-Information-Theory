"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation
from decoder_sc import (
    _all_filled,
    _leftdown,
    _rightdown,
    _up,
    _get_up_bit,
    _get_right_bit,
    _get_left_bit,
    _get_right_llr,
    _get_left_llr,
    _as_frozen_mask,
)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly = 0x07
        shift = 8
    elif crc_length == 16:
        poly = 0x8005
        shift = 16
    else:
        raise ValueError("crc_length must be 8 or 16")

    mask = (1 << crc_length) - 1
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(shift):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    if crc_length == 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


def _get_up_loc(bit_matrix):
    N = int(bit_matrix[0].size)
    n = int(math.log2(N))
    detect_array = bit_matrix[n]
    detect = -1
    for i in range(N):
        if not (detect_array[i] == 0 or detect_array[i] == 1):
            detect = i - 1
            break
    if detect == -1:
        return 0, 0
    if detect % 2 == 0:
        return n - 1, detect
    return n - 1, detect - 1


def _pm_update(llr_array, bit_array):
    pm = 0.0
    for i in range(llr_array.size):
        hard = 0 if llr_array[i] >= 0 else 1
        if hard != bit_array[i]:
            pm += abs(llr_array[i])
    return pm


def _sc_stepping_decoder(llr_matrix, bit_matrix, information_pos, frozen_val, split_pos):
    """译码至 split_pos 位判决完成"""
    N = int(bit_matrix[0].size)
    n = int(math.log2(N))
    information_pos = set(int(i) for i in information_pos)
    loc_row, loc_col = _get_up_loc(bit_matrix)
    position = [loc_row, loc_col, n, N]

    while not (bit_matrix[n][split_pos] == 0 or bit_matrix[n][split_pos] == 1):
        span = 2 ** (position[2] - position[0])
        up_llr = llr_matrix[position[0]][position[1] : position[1] + span]
        up_bit = bit_matrix[position[0]][position[1] : position[1] + span]
        left_llr = llr_matrix[position[0] + 1][position[1] : position[1] + span // 2]
        left_bit = bit_matrix[position[0] + 1][position[1] : position[1] + span // 2]
        right_llr = llr_matrix[position[0] + 1][position[1] + span // 2 : position[1] + span]
        right_bit = bit_matrix[position[0] + 1][position[1] + span // 2 : position[1] + span]

        if _all_filled(up_bit):
            position = _up(position)
        else:
            if _all_filled(right_bit):
                up_bit = _get_up_bit(left_bit, right_bit)
                bit_matrix[position[0]][position[1] : position[1] + span] = up_bit.copy()
            else:
                if _all_filled(right_llr):
                    if position[0] == position[2] - 1:
                        right_bit_pos = position[1] + span // 2
                        right_bit_val = _get_right_bit(
                            right_llr[0], information_pos, frozen_val, right_bit_pos
                        )
                        bit_matrix[position[0] + 1][
                            position[1] + span // 2 : position[1] + span
                        ] = right_bit_val
                    else:
                        position = _rightdown(position)
                else:
                    if _all_filled(left_bit):
                        right_llr = _get_right_llr(left_bit, up_llr)
                        llr_matrix[position[0] + 1][
                            position[1] + span // 2 : position[1] + span
                        ] = right_llr
                    else:
                        if not _all_filled(left_llr):
                            left_llr = _get_left_llr(up_llr)
                            llr_matrix[position[0] + 1][
                                position[1] : position[1] + span // 2
                            ] = left_llr
                        else:
                            if position[0] == position[2] - 1:
                                left_bit_pos = position[1]
                                left_bit_val = _get_left_bit(
                                    left_llr[0], information_pos, frozen_val, left_bit_pos
                                )
                                bit_matrix[position[0] + 1][
                                    position[1] : position[1] + span // 2
                                ] = left_bit_val
                            else:
                                position = _leftdown(position)

    return llr_matrix, bit_matrix


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = _as_frozen_mask(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        br = bit_reversal_permutation(N)
        y_llr = llr_ch[br]

        information_pos = list(self.info_indices)
        frozen_val = 0
        list_max = self.list_size

        llr_matrix = np.full((n + 1, N), np.nan)
        bit_matrix = np.full((n + 1, N), np.nan)
        llr_matrix[0] = y_llr

        llr_list = [llr_matrix]
        bit_list = [bit_matrix.copy()]
        pm_list = [0.0]
        split_pos = information_pos
        split_loc = 0
        l_now = 1

        while split_loc < len(split_pos):
            for i in range(l_now):
                llr_temp = llr_list[i]
                bit_temp = bit_list[i]
                pm_temp = pm_list[i]

                llr_out, bit_out = _sc_stepping_decoder(
                    llr_temp, bit_temp, information_pos, frozen_val, split_pos[split_loc]
                )
                llr_list[i] = llr_out
                bit_list[i] = bit_out

                prev = split_pos[split_loc - 1] + 1 if split_loc > 0 else 0
                curr = split_pos[split_loc] + 1
                pm_right = _pm_update(
                    llr_out[n][prev:curr], bit_out[n][prev:curr]
                )
                pm_list[i] = pm_temp + pm_right

                llr_list.append(llr_out.copy())
                bit_wrong = bit_out.copy()
                bit_wrong[n][split_pos[split_loc]] = 1 - bit_wrong[n][split_pos[split_loc]]
                bit_list.append(bit_wrong)
                pm_wrong = _pm_update(
                    llr_out[n][prev:curr], bit_wrong[n][prev:curr]
                )
                pm_list.append(pm_temp + pm_wrong)

            if l_now > list_max // 2:
                keep = np.argsort(pm_list)[:list_max]
                pm_list = [pm_list[i] for i in keep]
                llr_list = [llr_list[i] for i in keep]
                bit_list = [bit_list[i] for i in keep]

            l_now = len(pm_list)
            split_loc += 1

        if split_pos[-1] != N - 1:
            for i in range(l_now):
                llr_temp = llr_list[i]
                bit_temp = bit_list[i]
                pm_temp = pm_list[i]
                llr_out, bit_out = _sc_stepping_decoder(
                    llr_temp, bit_temp, information_pos, frozen_val, N - 1
                )
                llr_list[i] = llr_out
                bit_list[i] = bit_out
                prev = split_pos[-1] + 1
                pm_list[i] = pm_temp + _pm_update(
                    llr_out[n][prev:N], bit_out[n][prev:N]
                )

        order = np.argsort(pm_list)
        best_u = None
        best_pm = pm_list[order[0]]

        for idx in order:
            u_hat = bit_list[idx][n].astype(int)
            if self.crc_length > 0:
                if crc_check(u_hat[self.info_indices], self.crc_length):
                    best_u = u_hat
                    best_pm = pm_list[idx]
                    break
            else:
                best_u = u_hat
                best_pm = pm_list[idx]
                break

        if best_u is None:
            best_u = bit_list[order[0]][n].astype(int)

        return best_u, best_pm
