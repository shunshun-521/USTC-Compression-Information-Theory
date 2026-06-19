"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _all_filled,
    _get_left_llr,
    _get_right_llr,
    _get_up_bit,
    _get_up_loc,
    _leftdown,
    _rightdown,
    _sc_decode_tree,
    _up,
    reorder_channel_llr,
    sc_decode,
)
from utils import crc_check, crc_encode

__all__ = ["SCLDecoder", "crc_encode", "crc_check"]


def _path_metric(llr_array, bit_array):
    pm = 0.0
    for llr, bit in zip(llr_array, bit_array):
        hard = 0 if llr >= 0 else 1
        if hard != bit:
            pm += abs(llr)
    return pm


def _sc_step_until(llr_matrix, bit_matrix, information_pos, frozen_value, split_pos):
    """译码至 split_pos 并完成该位判决。"""
    N = bit_matrix.shape[1]
    n = int(math.log2(N))
    info_set = set(information_pos)
    loc = _get_up_loc(bit_matrix[n])
    position = [loc[0], loc[1], n, N]

    while bit_matrix[n][split_pos] not in (0, 1):
        span = 2 ** (position[2] - position[0])
        up_llr = llr_matrix[position[0]][position[1]:position[1] + span]
        up_bit = bit_matrix[position[0]][position[1]:position[1] + span]
        half = span // 2
        left_llr = llr_matrix[position[0] + 1][position[1]:position[1] + half]
        left_bit = bit_matrix[position[0] + 1][position[1]:position[1] + half]
        right_llr = llr_matrix[position[0] + 1][position[1] + half:position[1] + span]
        right_bit = bit_matrix[position[0] + 1][position[1] + half:position[1] + span]

        if _all_filled(up_bit):
            position = _up(position)
        elif _all_filled(right_bit):
            up_bit = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0]][position[1]:position[1] + span] = up_bit.copy()
        elif _all_filled(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                bit = (0 if right_llr[0] > 0 else 1) if right_bit_pos in info_set else frozen_value
                bit_matrix[position[0] + 1][position[1] + half:position[1] + span] = bit
            else:
                position = _rightdown(position)
        elif _all_filled(left_bit):
            right_llr = _get_right_llr(left_bit, up_llr)
            llr_matrix[position[0] + 1][position[1] + half:position[1] + span] = right_llr
        elif not _all_filled(left_llr):
            left_llr = _get_left_llr(up_llr)
            llr_matrix[position[0] + 1][position[1]:position[1] + half] = left_llr
        elif position[0] == position[2] - 1:
            left_bit_pos = position[1]
            bit = (0 if left_llr[0] >= 0 else 1) if left_bit_pos in info_set else frozen_value
            bit_matrix[position[0] + 1][position[1]:position[1] + half] = bit
        else:
            position = _leftdown(position)

    return llr_matrix, bit_matrix


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.information_pos = list(np.where(self.frozen_bits == 0)[0])

    def decode(self, llr_ch):
        if self.list_size == 1:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr = reorder_channel_llr(llr_ch)
        N, n = self.N, self.n
        info_pos = self.information_pos

        llr_list = [np.full((n + 1, N), np.nan)]
        bit_list = [np.full((n + 1, N), np.nan)]
        llr_list[0][0] = llr
        pm_list = [0.0]
        l_now = 1

        for split_loc, target in enumerate(info_pos):
            prev = info_pos[split_loc - 1] if split_loc > 0 else -1
            new_llr, new_bit, new_pm = [], [], []

            for idx in range(l_now):
                lm, bm, pm = llr_list[idx].copy(), bit_list[idx].copy(), pm_list[idx]
                lm, bm = _sc_step_until(lm, bm, info_pos, 0, target)
                seg_llr = lm[n][prev + 1:target + 1]
                seg_bit = bm[n][prev + 1:target + 1]

                new_llr.append(lm)
                new_bit.append(bm)
                new_pm.append(pm + _path_metric(seg_llr, seg_bit))

                bm2 = bm.copy()
                bm2[n][target] = 1 - bm2[n][target]
                seg_bit2 = bm2[n][prev + 1:target + 1]
                new_llr.append(lm.copy())
                new_bit.append(bm2)
                new_pm.append(pm + _path_metric(seg_llr, seg_bit2))

            order = np.argsort(new_pm)
            keep = order[: self.list_size]
            llr_list = [new_llr[i] for i in keep]
            bit_list = [new_bit[i] for i in keep]
            pm_list = [new_pm[i] for i in keep]
            l_now = len(pm_list)

        if info_pos and info_pos[-1] != N - 1:
            for idx in range(l_now):
                lm, bm, pm = llr_list[idx], bit_list[idx], pm_list[idx]
                lm, bm = _sc_step_until(lm, bm, info_pos, 0, N - 1)
                prev = info_pos[-1]
                pm_list[idx] = pm + _path_metric(lm[n][prev + 1:N], bm[n][prev + 1:N])
                llr_list[idx] = lm
                bit_list[idx] = bm

        candidates = [(pm, bit_list[i][n].astype(int)) for i, pm in enumerate(pm_list)]
        if self.crc_length > 0:
            valid = [(pm, u) for pm, u in candidates if crc_check(u[self.information_pos], self.crc_length)]
            if valid:
                candidates = valid

        best_pm, u_hat = min(candidates, key=lambda x: x[0])
        return u_hat, best_pm
