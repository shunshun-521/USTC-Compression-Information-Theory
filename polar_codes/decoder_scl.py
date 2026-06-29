"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    _prepare_llr, _all_num, _leftdown, _rightdown, _up,
    _get_up_bit, _get_right_bit, _get_right_llr,
    _get_left_bit, _get_left_llr, f_operation,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_step(reg, bit, crc_length, poly):
    mask = (1 << crc_length) - 1
    reg ^= (int(bit) << (crc_length - 1))
    if reg & (1 << (crc_length - 1)):
        reg = ((reg << 1) ^ poly) & mask
    else:
        reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for bit in info_bits:
        reg = _crc_step(reg, bit, crc_length, poly)
    crc_bits = np.array(
        [(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for bit in bits:
        reg = _crc_step(reg, bit, crc_length, poly)
    return reg == 0


def _get_up_loc(bit_matrix, n, N):
    detect_array = bit_matrix[n]
    detect = -1
    for i in range(N):
        if detect_array[i] != 0 and detect_array[i] != 1:
            detect = i - 1
            break
    if detect % 2 == 0:
        return [n - 1, detect]
    return [n - 1, detect - 1] if detect > 0 else [0, 0]


def _sc_step(llr_matrix, bit_matrix, info_set, frozen_bit, split_pos):
    """SC 译码运行至 split_pos 位置"""
    N = bit_matrix.shape[1]
    n = int(math.log2(N))
    info_set = set(info_set)
    loc = _get_up_loc(bit_matrix, n, N)
    position = [loc[0], loc[1], n, N]

    while bit_matrix[n][split_pos] != 0 and bit_matrix[n][split_pos] != 1:
        span = 2 ** (position[2] - position[0])
        p0, p1 = position[0], position[1]
        up_llr = llr_matrix[p0][p1:p1 + span]
        up_bit = bit_matrix[p0][p1:p1 + span]
        left_llr = llr_matrix[p0 + 1][p1:p1 + span // 2]
        left_bit = bit_matrix[p0 + 1][p1:p1 + span // 2]
        right_llr = llr_matrix[p0 + 1][p1 + span // 2:p1 + span]
        right_bit = bit_matrix[p0 + 1][p1 + span // 2:p1 + span]

        if _all_num(up_bit):
            position = _up(position)
        elif _all_num(right_bit):
            up_bit = _get_up_bit(left_bit, right_bit)
            bit_matrix[p0][p1:p1 + span] = up_bit.copy()
        elif _all_num(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                right_bit = _get_right_bit(right_llr, info_set, frozen_bit, right_bit_pos)
                bit_matrix[p0 + 1][p1 + span // 2:p1 + span] = right_bit
            else:
                position = _rightdown(position)
        elif _all_num(left_bit):
            right_llr = _get_right_llr(left_bit, up_llr)
            llr_matrix[p0 + 1][p1 + span // 2:p1 + span] = right_llr
        elif not _all_num(left_llr):
            left_llr = _get_left_llr(up_llr)
            llr_matrix[p0 + 1][p1:p1 + span // 2] = left_llr
        elif position[0] == position[2] - 1:
            left_bit_pos = position[1]
            left_bit = _get_left_bit(left_llr, info_set, frozen_bit, left_bit_pos)
            bit_matrix[p0 + 1][p1:p1 + span // 2] = left_bit
        else:
            position = _leftdown(position)

    return llr_matrix, bit_matrix


def _pm_update(llr_slice, bit_slice):
  pm = 0.0
  for llr, bit in zip(llr_slice, bit_slice):
      hard = 0 if llr >= 0 else 1
      if bit != hard:
          pm += abs(llr)
  return pm


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_set = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm"""
        N = self.N
        n = self.n
        y_llr = _prepare_llr(llr_ch, N)
        split_pos = list(self.info_set)
        if len(split_pos) == 0:
            return np.zeros(N, dtype=int), 0.0

        llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
        bit_matrix = np.full((n + 1, N), np.nan)
        llr_matrix[0] = y_llr

        llr_list = [llr_matrix.copy()]
        bit_list = [bit_matrix.copy()]
        pm_list = [0.0]
        split_loc = 0

        while split_loc < len(split_pos):
            l_now = len(pm_list)
            new_llr, new_bit, new_pm = [], [], []

            prev_start = split_pos[split_loc - 1] + 1 if split_loc > 0 else 0
            cur_pos = split_pos[split_loc]

            for i in range(l_now):
                lm = llr_list[i].copy()
                bm = bit_list[i].copy()
                lm, bm = _sc_step(lm, bm, self.info_set, 0, cur_pos)

                llr_seg = lm[n][prev_start:cur_pos + 1]
                bit_seg = bm[n][prev_start:cur_pos + 1].astype(int)

                new_llr.append(lm)
                new_bit.append(bm)
                new_pm.append(pm_list[i] + _pm_update(llr_seg, bit_seg))

                bm_wrong = bm.copy()
                bm_wrong[n][cur_pos] = 1 - int(bm_wrong[n][cur_pos])
                bit_seg_w = bm_wrong[n][prev_start:cur_pos + 1].astype(int)
                new_llr.append(lm.copy())
                new_bit.append(bm_wrong)
                new_pm.append(pm_list[i] + _pm_update(llr_seg, bit_seg_w))

            order = np.argsort(new_pm)
            keep = order[:self.list_size]
            llr_list = [new_llr[i] for i in keep]
            bit_list = [new_bit[i] for i in keep]
            pm_list = [new_pm[i] for i in keep]
            split_loc += 1

        if split_pos[-1] != N - 1:
            for i in range(len(pm_list)):
                lm = llr_list[i].copy()
                bm = bit_list[i].copy()
                lm, bm = _sc_step(lm, bm, self.info_set, 0, N - 1)
                llr_list[i] = lm
                bit_list[i] = bm
                prev_start = split_pos[-1] + 1
                pm_list[i] += _pm_update(
                    lm[n][prev_start:N], bm[n][prev_start:N].astype(int)
                )

        order = np.argsort(pm_list)
        best_u = None
        best_pm = pm_list[order[0]]

        if self.crc_length > 0:
            for idx in order:
                u_cand = bit_list[idx][n].astype(int)
                if crc_check(u_cand, self.crc_length):
                    return u_cand, pm_list[idx]

        best_u = bit_list[order[0]][n].astype(int)
        return best_u, best_pm
