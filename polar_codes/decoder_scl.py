"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    _sc_tree_decode, _all_computed, _up, _leftdown, _rightdown,
    _get_up_bit, _get_right_llr, _get_left_llr, f_operation, g_operation
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_encode_bits(info_bits, poly, crc_length):
    reg = 0
    for bit in info_bits:
        reg = ((reg << 1) | int(bit))
        if (reg >> crc_length) & 1:
            reg ^= poly
    for _ in range(crc_length):
        reg <<= 1
        if (reg >> crc_length) & 1:
            reg ^= poly
    return reg & ((1 << crc_length) - 1)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_encode_bits(info_bits, poly, crc_length)
    crc_bits = np.array([(remainder >> (crc_length - 1 - i)) & 1
                         for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_encode_bits(bits, poly, crc_length)
    return remainder == 0


def _get_up_loc(bit_matrix):
    N = bit_matrix.shape[1]
    n = int(np.log2(N))
    detect = -1
    for i in range(N):
        if np.isnan(bit_matrix[n, i]):
            detect = i - 1
            break
    if detect % 2 == 0:
        loc_row, loc_col = n - 1, max(detect, 0)
    else:
        loc_row, loc_col = n - 1, max(detect - 1, 0)
    if detect == -1:
        loc_row, loc_col = 0, 0
    return [loc_row, loc_col]


def _sc_step_to_split(llr_matrix, bit_matrix, information_pos, frozen_bit, split_pos):
    """SC 译码到 split_pos 位置"""
    N = bit_matrix.shape[1]
    n = int(np.log2(N))
    loc = _get_up_loc(bit_matrix)
    position = [loc[0], loc[1], n, N]

    while np.isnan(bit_matrix[n, split_pos]):
        span = 2 ** (position[2] - position[0])
        p0, p1 = position[0], position[1]
        up_llr = llr_matrix[p0][p1:p1 + span]
        up_bit = bit_matrix[p0][p1:p1 + span]
        half = span // 2
        left_llr = llr_matrix[p0 + 1][p1:p1 + half]
        left_bit = bit_matrix[p0 + 1][p1:p1 + half]
        right_llr = llr_matrix[p0 + 1][p1 + half:p1 + span]
        right_bit = bit_matrix[p0 + 1][p1 + half:p1 + span]

        if _all_computed(up_bit):
            position = _up(position)
        elif _all_computed(right_bit):
            up_bit = _get_up_bit(left_bit, right_bit)
            bit_matrix[p0][p1:p1 + span] = up_bit
        elif not _all_computed(right_llr):
            if _all_computed(left_bit):
                right_llr = _get_right_llr(left_bit, up_llr)
                llr_matrix[p0 + 1][p1 + half:p1 + span] = right_llr
            elif not _all_computed(left_llr):
                left_llr = _get_left_llr(up_llr)
                llr_matrix[p0 + 1][p1:p1 + half] = left_llr
            elif position[0] == position[2] - 1:
                left_bit_pos = p1
                if left_bit_pos in information_pos:
                    bit_matrix[p0 + 1][p1] = 0 if left_llr[0] >= 0 else 1
                else:
                    bit_matrix[p0 + 1][p1] = frozen_bit
            else:
                position = _leftdown(position)
        elif position[0] == position[2] - 1:
            right_bit_pos = p1 + 1
            if right_bit_pos in information_pos:
                bit_matrix[p0 + 1][p1 + half] = 0 if right_llr[0] >= 0 else 1
            else:
                bit_matrix[p0 + 1][p1 + half] = frozen_bit
        else:
            position = _rightdown(position)

    return llr_matrix, bit_matrix


def _pm_update(llr_array, bit_array):
    pm = 0.0
    for i in range(len(llr_array)):
        expected = 1 - 2 * bit_array[i]
        if np.sign(llr_array[i]) != expected:
            pm += abs(llr_array[i])
    return pm


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.rev = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)"""
        N = self.N
        n = self.n
        llr_br = llr_ch[self.rev]
        info_pos = self.info_indices
        frozen_bit = 0

        if self.list_size == 1 and self.crc_length == 0:
            u_hat = _sc_tree_decode(llr_br, info_pos, frozen_bit)
            return u_hat, 0.0

        llr_matrix = np.full((n + 1, N), np.nan)
        bit_matrix = np.full((n + 1, N), np.nan)
        llr_matrix[0] = llr_br

        llr_list = [llr_matrix.copy()]
        bit_list = [bit_matrix.copy()]
        pm_list = [0.0]
        split_pos = list(info_pos)
        split_loc = 0
        l_now = 1

        while split_loc < len(split_pos):
            new_llr_list = []
            new_bit_list = []
            new_pm_list = []

            for i in range(l_now):
                lm, bm = _sc_step_to_split(
                    llr_list[i].copy(), bit_list[i].copy(),
                    info_pos, frozen_bit, split_pos[split_loc])

                prev_end = split_pos[split_loc - 1] + 1 if split_loc > 0 else 0
                cur_end = split_pos[split_loc] + 1
                llr_seg = lm[n][prev_end:cur_end]
                bit_seg = bm[n][prev_end:cur_end].astype(int)

                pm_base = pm_list[i]
                new_pm_list.append(pm_base + _pm_update(llr_seg, bit_seg))
                new_llr_list.append(lm)
                new_bit_list.append(bm)

                bm_wrong = bm.copy()
                bm_wrong[n][split_pos[split_loc]] = 1 - bm[n][split_pos[split_loc]]
                wrong_seg = bm_wrong[n][prev_end:cur_end].astype(int)
                new_pm_list.append(pm_base + _pm_update(llr_seg, wrong_seg))
                new_llr_list.append(lm.copy())
                new_bit_list.append(bm_wrong)

            order = np.argsort(new_pm_list)
            keep = order[:self.list_size]
            llr_list = [new_llr_list[i] for i in keep]
            bit_list = [new_bit_list[i] for i in keep]
            pm_list = [new_pm_list[i] for i in keep]
            l_now = len(pm_list)
            split_loc += 1

        if split_pos and split_pos[-1] != N - 1:
            for i in range(l_now):
                lm, bm = _sc_step_to_split(
                    llr_list[i].copy(), bit_list[i].copy(),
                    info_pos, frozen_bit, N - 1)
                llr_list[i] = lm
                bit_list[i] = bm
                prev_end = split_pos[-1] + 1
                pm_list[i] += _pm_update(
                    lm[n][prev_end:N], bm[n][prev_end:N].astype(int))

        order = np.argsort(pm_list)

        if self.crc_length > 0:
            for idx in order:
                u_d = bit_list[idx][n].astype(int)
                if crc_check(u_d[info_pos], self.crc_length):
                    return u_d, pm_list[idx]
            best = order[0]
        else:
            best = order[0]

        return bit_list[best][n].astype(int), pm_list[best]
