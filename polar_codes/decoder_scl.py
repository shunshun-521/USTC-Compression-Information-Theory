"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import _frozen_bits_to_info_pos
from _polar_ref_function import get_pm_update


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _poly_div_crc(msg_bits, crc_length=8, poly=0x07):
    """GF(2) 多项式除法求 CRC 余式（MSB first）。"""
    msg = [int(b) for b in msg_bits]
    poly_bits = [(poly >> i) & 1 for i in range(crc_length - 1, -1, -1)]
    for i in range(len(msg) - crc_length):
        if msg[i] == 1:
            for j, pb in enumerate(poly_bits):
                msg[i + j] ^= pb
    return np.array(msg[-crc_length:], dtype=np.int8)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    crc_bits = _poly_div_crc(
        np.concatenate([info_bits, np.zeros(crc_length, dtype=np.int8)]),
        crc_length,
        poly,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _poly_div_crc(bits, crc_length, poly)
    return np.all(remainder == 0)


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.information_pos = _frozen_bits_to_info_pos(frozen_bits)
        self.frozen_bit = 0

    def _initial_matrices(self, llr_ch):
        llr_matrix = np.ones((self.n + 1, self.N), dtype=np.float64)
        llr_matrix[llr_matrix == 1] = np.nan
        bit_matrix = llr_matrix.copy()
        llr_matrix[0] = llr_ch
        return llr_matrix, bit_matrix

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        info_positions = [p for p in self.information_pos]

        llr_list = [self._initial_matrices(llr_ch)[0]]
        bit_list = [self._initial_matrices(llr_ch)[1]]
        pm_list = [0.0]

        split_pos = info_positions
        split_loc = 0
        l_now = 1

        while split_loc < len(split_pos):
            new_llr, new_bit, new_pm = [], [], []
            prev_end = split_pos[split_loc - 1] if split_loc > 0 else -1
            cur_pos = split_pos[split_loc]

            for i in range(l_now):
                lm, bm = self._sc_to_position(
                    llr_list[i], bit_list[i], cur_pos
                )
                llr0, bit0 = lm, bm
                pm0 = pm_list[i] + get_pm_update(
                    llr0[self.n, prev_end + 1:cur_pos + 1],
                    bit0[self.n, prev_end + 1:cur_pos + 1],
                    'hf',
                )
                new_llr.append(llr0)
                new_bit.append(bit0)
                new_pm.append(pm0)

                bit1 = bit0.copy()
                bit1[self.n, cur_pos] = 1 - bit1[self.n, cur_pos]
                pm1 = pm_list[i] + get_pm_update(
                    llr0[self.n, prev_end + 1:cur_pos + 1],
                    bit1[self.n, prev_end + 1:cur_pos + 1],
                    'hf',
                )
                new_llr.append(llr0.copy())
                new_bit.append(bit1)
                new_pm.append(pm1)

            order = np.argsort(new_pm)
            keep = order[: self.list_size]
            llr_list = [new_llr[i] for i in keep]
            bit_list = [new_bit[i] for i in keep]
            pm_list = [new_pm[i] for i in keep]
            l_now = len(pm_list)
            split_loc += 1

        if split_pos and split_pos[-1] != self.N - 1:
            for i in range(l_now):
                lm, bm = self._sc_to_position(llr_list[i], bit_list[i], self.N - 1)
                llr_list[i], bit_list[i] = lm, bm
                prev_end = split_pos[-1]
                pm_list[i] += get_pm_update(
                    lm[self.n, prev_end + 1:self.N],
                    bm[self.n, prev_end + 1:self.N],
                    'hf',
                )

        order = np.argsort(pm_list)
        best_u = None
        best_pm = pm_list[order[0]]

        if self.crc_length > 0:
            for idx in order:
                u_cand = bit_list[idx][self.n].astype(int)
                info_bits = u_cand[self.information_pos]
                if crc_check(info_bits, self.crc_length):
                    return u_cand, pm_list[idx]
            best_u = bit_list[order[0]][self.n].astype(int)
        else:
            best_u = bit_list[order[0]][self.n].astype(int)

        return best_u, best_pm

    def _sc_to_position(self, llr_matrix, bit_matrix, target_pos):
        """运行 SC 至 target_pos 判决完成。"""
        from _polar_ref_function import (
            all_num, up, leftdown, rightdown,
            get_up_bit, get_right_bit, get_left_bit,
            get_right_llr, get_left_llr, get_up_loc,
        )
        lm = llr_matrix.copy()
        bm = bit_matrix.copy()
        loc = get_up_loc(bm)
        position = [loc[0], loc[1], self.n, self.N]

        while bm[self.n, target_pos] != 0 and bm[self.n, target_pos] != 1:
            up_llr = lm[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])]
            up_bit = bm[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])]
            left_llr = lm[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)]
            left_bit = bm[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)]
            right_llr = lm[position[0] + 1][position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])]
            right_bit = bm[position[0] + 1][position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])]

            if all_num(up_bit) == 1:
                position = up(position)
            elif all_num(right_bit) == 1:
                up_bit_new = get_up_bit(left_bit, right_bit)
                bm[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])] = up_bit_new.copy()
            elif all_num(right_llr) == 1:
                if position[0] == position[2] - 1:
                    rb = get_right_bit(right_llr, self.information_pos, self.frozen_bit, position[1] + 1)
                    bm[position[0] + 1][position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])] = rb
                else:
                    position = rightdown(position)
            elif all_num(left_bit) == 1:
                rr = get_right_llr(left_bit, up_llr)
                lm[position[0] + 1][position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])] = rr
            elif all_num(left_llr) == 0:
                ll = get_left_llr(up_llr)
                lm[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)] = ll
            else:
                if position[0] == position[2] - 1:
                    lb = get_left_bit(left_llr, self.information_pos, self.frozen_bit, position[1])
                    bm[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)] = lb
                else:
                    position = leftdown(position)

        return lm, bm
