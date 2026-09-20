"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math

from decoder_sc import sc_decode, _get_bit


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_poly_bits(crc_length):
    if crc_length == 8:
        loc = [8, 2, 1, 0]
    elif crc_length == 16:
        loc = [16, 15, 2, 0]
    else:
        raise ValueError("crc_length must be 8 or 16")
    p = [0] * (crc_length + 1)
    for i in loc:
        p[i] = 1
    return p[::-1]


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = list(np.asarray(info_bits, dtype=int))
    p = _crc_poly_bits(crc_length)
    r = crc_length
    work = info_bits + [0] * r
    times = len(info_bits)
    for i in range(times):
        if work[i] == 1:
            for j in range(r + 1):
                work[i + j] ^= p[j]
    check = work[-r:]
    return np.array(info_bits + check, dtype=int)


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = list(np.asarray(bits, dtype=int))
    if len(bits) < crc_length:
        return False
    info = bits[:-crc_length]
    expected = crc_encode(info, crc_length)
    return list(expected) == bits


def _pm_update(llr_val, bit):
    u_from_llr = 0 if llr_val >= 0 else 1
    return 0.0 if bit == u_from_llr else abs(llr_val)


def _sc_step_to(llr_matrix, bit_matrix, info_pos, frozen_bits, stop_pos):
    """SC 逐步译码到位置 stop_pos（含）"""
    from decoder_sc import (
        _all_filled, _get_left_llr, _get_right_llr, _get_up_bit,
        _leftdown, _rightdown, _up,
    )

    N = bit_matrix.shape[1]
    n = int(math.log2(N))
    position = [0, 0, n, N]

    while not (bit_matrix[n][stop_pos] == 0 or bit_matrix[n][stop_pos] == 1):
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
            bit_matrix[position[0]][position[1]:position[1] + span] = up_bit
        elif not _all_filled(right_llr):
            if _all_filled(left_bit):
                right_llr = _get_right_llr(left_bit, up_llr)
                llr_matrix[position[0] + 1][position[1] + half:position[1] + span] = right_llr
            elif not _all_filled(left_llr):
                left_llr = _get_left_llr(up_llr)
                llr_matrix[position[0] + 1][position[1]:position[1] + half] = left_llr
            elif position[0] == position[2] - 1:
                left_bit_pos = position[1]
                bit_matrix[position[0] + 1][position[1]] = _get_bit(
                    left_llr[0], left_bit_pos in info_pos
                )
            else:
                position = _leftdown(position)
        elif position[0] == position[2] - 1:
            right_bit_pos = position[1] + 1
            bit_matrix[position[0] + 1][position[1] + half] = _get_bit(
                right_llr[0], right_bit_pos in info_pos
            )
        else:
            position = _rightdown(position)

    return llr_matrix, bit_matrix


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_pos = sorted(np.where(~self.frozen_bits)[0])

    def decode(self, llr_ch):
        y_llr = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        if self.list_size == 1:
            u_hat = sc_decode(y_llr, self.frozen_bits)
            return u_hat, 0.0

        llr_list = []
        bit_list = []
        pm_list = []

        llr0 = np.full((n + 1, N), np.nan)
        bit0 = np.full((n + 1, N), np.nan)
        llr0[0] = y_llr
        llr_list.append(llr0)
        bit_list.append(bit0)
        pm_list.append(0.0)

        split_positions = [p for p in self.info_pos]

        for split_idx, split_pos in enumerate(split_positions):
            new_llr, new_bit, new_pm = [], [], []
            prev_start = split_positions[split_idx - 1] if split_idx > 0 else -1

            for path_i in range(len(llr_list)):
                lm, bm = _sc_step_to(
                    llr_list[path_i].copy(), bit_list[path_i].copy(),
                    set(self.info_pos), self.frozen_bits, split_pos,
                )
                llr_at = lm[n][split_pos]
                bit0 = int(bm[n][split_pos])
                bit1 = 1 - bit0
                pm_base = pm_list[path_i]

                for bit in (bit0, bit1):
                    bm_c = bm.copy()
                    bm_c[n][split_pos] = bit
                    new_llr.append(lm.copy())
                    new_bit.append(bm_c)
                    new_pm.append(pm_base + _pm_update(llr_at, bit))

            order = np.argsort(new_pm)
            keep = order[: self.list_size]
            llr_list = [new_llr[i] for i in keep]
            bit_list = [new_bit[i] for i in keep]
            pm_list = [new_pm[i] for i in keep]

        last_pos = N - 1
        if split_positions and split_positions[-1] != last_pos:
            final_llr, final_bit, final_pm = [], [], []
            for path_i in range(len(llr_list)):
                lm, bm = _sc_step_to(
                    llr_list[path_i].copy(), bit_list[path_i].copy(),
                    set(self.info_pos), self.frozen_bits, last_pos,
                )
                final_llr.append(lm)
                final_bit.append(bm)
                final_pm.append(pm_list[path_i])
            llr_list, bit_list, pm_list = final_llr, final_bit, final_pm

        candidates = list(range(len(pm_list)))
        if self.crc_length > 0:
            valid = []
            for i in candidates:
                u = bit_list[i][n].astype(int)
                info_bits = u[self.info_pos]
                if crc_check(info_bits, self.crc_length):
                    valid.append(i)
            if valid:
                candidates = valid

        best = min(candidates, key=lambda i: pm_list[i])
        return bit_list[best][n].astype(int), pm_list[best]
