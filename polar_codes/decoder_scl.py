"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import math

import numpy as np

from decoder_sc import sc_decode, channel_llr_to_decoder, f_operation, g_operation
from _sc_backend import sc_decoder as _sc_core, all_num, up, leftdown, rightdown, get_up_bit, get_right_bit, get_left_bit, get_right_llr, get_left_llr
from utils import crc_encode, crc_check


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.info_idx = np.where(self.frozen_bits == 0)[0].tolist()
        self.list_size = list_size
        self.crc_length = crc_length

    def _pm_update(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def _sc_step_to(self, llr_matrix, bit_matrix, stop_pos):
        N, n = self.N, self.n
        information_pos = self.info_idx
        frozen_bit = 0
        position = [0, 0, n, N]
        while bit_matrix[n][stop_pos] != 0 and bit_matrix[n][stop_pos] != 1:
            up_llr = llr_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])]
            up_bit = bit_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])]
            left_llr = llr_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)]
            left_bit = bit_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)]
            right_llr = llr_matrix[position[0] + 1][
                position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])
            ]
            right_bit = bit_matrix[position[0] + 1][
                position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])
            ]
            if all_num(up_bit) == 1:
                position = up(position)
            elif all_num(right_bit) == 1:
                up_bit = get_up_bit(left_bit, right_bit)
                bit_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])] = up_bit.copy()
            elif all_num(right_llr) == 1:
                if position[0] == position[2] - 1:
                    pos = position[1] + 1
                    val = get_right_bit(right_llr, information_pos, frozen_bit, pos)
                    bit_matrix[position[0] + 1][
                        position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])
                    ] = val
                else:
                    position = rightdown(position)
            elif all_num(left_bit) == 1:
                right_llr = get_right_llr(left_bit, up_llr)
                llr_matrix[position[0] + 1][
                    position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])
                ] = right_llr
            elif all_num(left_llr) == 0:
                left_llr = get_left_llr(up_llr)
                llr_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)] = left_llr
            else:
                if position[0] == position[2] - 1:
                    pos = position[1]
                    val = get_left_bit(left_llr, information_pos, frozen_bit, pos)
                    bit_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)] = val
                else:
                    position = leftdown(position)
        return llr_matrix, bit_matrix

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr = channel_llr_to_decoder(np.asarray(llr_ch, dtype=np.float64))
        N, n = self.N, self.n
        base_llr = np.ones((n + 1, N))
        base_llr[:] = np.nan
        base_bit = base_llr.copy()
        base_llr[0] = llr

        paths = [{"llr": copy.deepcopy(base_llr), "bit": copy.deepcopy(base_bit), "pm": 0.0}]
        split_positions = self.info_idx
        prev = -1
        for split_pos in split_positions:
            new_paths = []
            for path in paths:
                llr_m, bit_m = self._sc_step_to(path["llr"], path["bit"], split_pos)
                leaf_llr = llr_m[n][split_pos]
                b0 = int(bit_m[n][split_pos])
                b1 = 1 - b0
                for b in (b0, b1):
                    p = {
                        "llr": copy.deepcopy(llr_m),
                        "bit": copy.deepcopy(bit_m),
                        "pm": path["pm"] + self._pm_update(leaf_llr, b),
                    }
                    p["bit"][n][split_pos] = b
                    new_paths.append(p)
            new_paths.sort(key=lambda x: x["pm"])
            paths = new_paths[: self.list_size]
            prev = split_pos

        if prev < N - 1:
            finalized = []
            for path in paths:
                llr_m, bit_m = self._sc_step_to(path["llr"], path["bit"], N - 1)
                finalized.append({"llr": llr_m, "bit": bit_m, "pm": path["pm"]})
            paths = finalized

        paths.sort(key=lambda x: x["pm"])
        if self.crc_length > 0:
            for path in paths:
                u_hat = path["bit"][n].astype(int)
                if crc_check(u_hat[self.info_idx], self.crc_length):
                    return u_hat, path["pm"]
        u_hat = paths[0]["bit"][n].astype(int)
        return u_hat, paths[0]["pm"]
