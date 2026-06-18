"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    _all_num,
    _get_left_bit,
    _get_right_bit,
    _left_llr,
    _leftdown,
    _right_llr,
    _rightdown,
    _sc_tree_decode,
    _up,
    _up_bit,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_poly(crc_length):
    if crc_length == 8:
        loc = [8, 2, 1, 0]
    elif crc_length == 16:
        loc = [16, 15, 2, 0]
    else:
        raise ValueError(f"Unsupported CRC length: {crc_length}")
    poly = [0] * (crc_length + 1)
    for i in loc:
        poly[i] = 1
    return poly[::-1]


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后（多项式长除法）"""
    info_bits = [int(b) for b in np.asarray(info_bits, dtype=int)]
    p = _crc_poly(crc_length)
    work = info_bits + [0] * crc_length
    times = len(info_bits)
    for i in range(times):
        if work[i] == 1:
            for j in range(crc_length + 1):
                work[i + j] ^= p[j]
    check_code = work[-crc_length:]
    return np.array(info_bits + check_code, dtype=int)


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = [int(b) for b in np.asarray(bits, dtype=int)]
    info = bits[:-crc_length]
    encoded = crc_encode(info, crc_length)
    return encoded.tolist() == bits


def _pm_update(llr_array, bit_array):
    pm = 0.0
    for llr, bit in zip(llr_array, bit_array):
        hard = 0 if llr >= 0 else 1
        if hard != bit:
            pm += abs(llr)
    return pm


def _get_up_loc(bit_matrix):
    N = bit_matrix.shape[1]
    n = int(math.log2(N))
    detect_array = bit_matrix[n]
    detect = -1
    for i in range(N):
        if not (detect_array[i] == 0 or detect_array[i] == 1):
            detect = i - 1
            break
    if detect == -1:
        return [0, 0]
    if detect % 2 == 0:
        return [n - 1, detect]
    return [n - 1, detect - 1]


def _sc_step_to(llr_matrix, bit_matrix, information_pos, frozen_bit, split_pos):
    """SC 译码到 split_pos 位（含）"""
    N = bit_matrix.shape[1]
    n = int(math.log2(N))
    loc = _get_up_loc(bit_matrix)
    position = [loc[0], loc[1], n, N]

    while bit_matrix[n, split_pos] != 0 and bit_matrix[n, split_pos] != 1:
        span = 2 ** (position[2] - position[0])
        up_llr = llr_matrix[position[0], position[1] : position[1] + span]
        up_bit = bit_matrix[position[0], position[1] : position[1] + span]
        half = span // 2
        left_llr = llr_matrix[position[0] + 1, position[1] : position[1] + half]
        left_bit = bit_matrix[position[0] + 1, position[1] : position[1] + half]
        right_llr = llr_matrix[position[0] + 1, position[1] + half : position[1] + span]
        right_bit = bit_matrix[position[0] + 1, position[1] + half : position[1] + span]

        if _all_num(up_bit) == 1:
            position = _up(position)
        elif _all_num(right_bit) == 1:
            merged = _up_bit(left_bit, right_bit)
            bit_matrix[position[0], position[1] : position[1] + span] = merged
        elif _all_num(right_llr) == 1:
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + half
                val = _get_right_bit(
                    right_llr[0], information_pos, frozen_bit, right_bit_pos
                )
                bit_matrix[position[0] + 1, position[1] + half : position[1] + span] = (
                    val
                )
            else:
                position = _rightdown(position)
        elif _all_num(left_bit) == 1:
            right_llr_new = _right_llr(left_bit, up_llr)
            llr_matrix[position[0] + 1, position[1] + half : position[1] + span] = (
                right_llr_new
            )
        elif _all_num(left_llr) == 0:
            left_llr_new = _left_llr(up_llr)
            llr_matrix[position[0] + 1, position[1] : position[1] + half] = left_llr_new
        else:
            if position[0] == position[2] - 1:
                left_bit_pos = position[1]
                val = _get_left_bit(
                    left_llr[0], information_pos, frozen_bit, left_bit_pos
                )
                bit_matrix[position[0] + 1, position[1] : position[1] + half] = val
            else:
                position = _leftdown(position)

    return llr_matrix, bit_matrix


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.information_pos = list(np.where(~self.frozen_bits)[0])

    def decode(self, llr_ch):
        N, n = self.N, self.n
        L = self.list_size
        y_llr = np.asarray(llr_ch, dtype=np.float64)

        llr_matrix = np.ones((n + 1, N), dtype=np.float64)
        llr_matrix[llr_matrix == 1] = np.nan
        bit_matrix = llr_matrix.copy()
        llr_matrix[0] = y_llr

        llr_list = [llr_matrix.copy()]
        bit_list = [bit_matrix.copy()]
        pm_list = [0.0]

        split_pos = [p for p in self.information_pos]
        split_loc = 0
        l_now = 1

        while split_loc < len(split_pos):
            prev = split_pos[split_loc - 1] if split_loc > 0 else -1
            cur = split_pos[split_loc]
            new_llr_list = []
            new_bit_list = []
            new_pm_list = []

            for i in range(l_now):
                lm, bm = _sc_step_to(
                    llr_list[i].copy(),
                    bit_list[i].copy(),
                    self.information_pos,
                    0,
                    cur,
                )
                pm_base = pm_list[i]
                llr_seg = lm[n, prev + 1 : cur + 1]
                bit_seg = bm[n, prev + 1 : cur + 1].astype(int)

                new_llr_list.append(lm)
                new_bit_list.append(bm)
                new_pm_list.append(pm_base + _pm_update(llr_seg, bit_seg))

                bm_wrong = bm.copy()
                bm_wrong[n, cur] = 1 - bm_wrong[n, cur]
                bit_seg_w = bm_wrong[n, prev + 1 : cur + 1].astype(int)
                new_llr_list.append(lm.copy())
                new_bit_list.append(bm_wrong)
                new_pm_list.append(pm_base + _pm_update(llr_seg, bit_seg_w))

            order = np.argsort(new_pm_list)
            keep = order[:L]
            llr_list = [new_llr_list[i] for i in keep]
            bit_list = [new_bit_list[i] for i in keep]
            pm_list = [new_pm_list[i] for i in keep]
            l_now = len(pm_list)
            split_loc += 1

        if split_pos and split_pos[-1] != N - 1:
            for i in range(l_now):
                lm, bm = _sc_step_to(
                    llr_list[i].copy(),
                    bit_list[i].copy(),
                    self.information_pos,
                    0,
                    N - 1,
                )
                llr_list[i] = lm
                bit_list[i] = bm

        candidates = []
        for i in range(l_now):
            u_hat = bit_list[i][n].astype(int)
            pm = pm_list[i]
            if self.crc_length > 0:
                payload = u_hat[self.information_pos]
                if crc_check(payload, self.crc_length):
                    candidates.append((pm, u_hat, True))
                else:
                    candidates.append((pm, u_hat, False))
            else:
                candidates.append((pm, u_hat, True))

        valid = [c for c in candidates if c[2]]
        if valid:
            best = min(valid, key=lambda x: x[0])
        else:
            best = min(candidates, key=lambda x: x[0])

        return best[1], best[0]
