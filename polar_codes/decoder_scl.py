"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    _all_decided,
    _combine_bits,
    _frozen_bits_to_info_pos,
    f_operation,
    g_operation,
    sc_decode,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return [1, 0, 0, 0, 0, 0, 1, 1, 1]
    if crc_length == 16:
        return [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info = [int(b) for b in info_bits]
    poly = _crc_poly(crc_length)
    padded = info + [0] * crc_length
    for i in range(len(info)):
        if padded[i] == 1:
            for j in range(len(poly)):
                padded[i + j] ^= poly[j]
    check = padded[len(info) : len(info) + crc_length]
    return np.array(info + check, dtype=int)


def crc_check(bits, crc_length=8):
    """检验 CRC，返回 True/False"""
    bits = [int(b) for b in bits]
    info = bits[:-crc_length]
    return np.array_equal(crc_encode(info, crc_length), bits)


def _get_up_loc(bit_matrix, n):
    detect_array = bit_matrix[n]
    detect = -1
    for i in range(len(detect_array)):
        if detect_array[i] in (0, 1):
            continue
        detect = i - 1
        break
    if detect == -1:
        return [0, 0]
    if detect % 2 == 0:
        return [n - 1, detect]
    return [n - 1, detect - 1]


def sc_stepping_decoder(llr_matrix, bit_matrix, information_pos, frozen_bit, split_pos, n, N):
    """SC 译码推进到 split_pos 判决完成"""
    loc = _get_up_loc(bit_matrix, n)
    position = [loc[0], loc[1], n, N]

    while bit_matrix[n][split_pos] not in (0, 1):
        span = 2 ** (position[2] - position[0])
        up_llr = llr_matrix[position[0], position[1] : position[1] + span]
        half = span // 2
        left_llr = llr_matrix[position[0] + 1, position[1] : position[1] + half]
        left_bit = bit_matrix[position[0] + 1, position[1] : position[1] + half]
        right_llr = llr_matrix[position[0] + 1, position[1] + half : position[1] + span]
        right_bit = bit_matrix[position[0] + 1, position[1] + half : position[1] + span]

        if _all_decided(bit_matrix[position[0], position[1] : position[1] + span]):
            p0 = position[0] - 1
            p1 = int(
                np.floor(position[1] / (2 ** (position[2] - position[0] + 1)))
                * (2 ** (position[2] - position[0] + 1))
            )
            position = [p0, p1, position[2], position[3]]
        elif _all_decided(right_bit):
            bit_matrix[position[0], position[1] : position[1] + span] = _combine_bits(
                left_bit, right_bit
            )
        elif _all_decided(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                if right_bit_pos in information_pos:
                    val = 0 if right_llr[0] >= 0 else 1
                else:
                    val = frozen_bit
                bit_matrix[position[0] + 1, position[1] + half : position[1] + span] = val
            else:
                position = [
                    position[0] + 1,
                    position[1] + 2 ** (position[2] - 1 - position[0]),
                    position[2],
                    position[3],
                ]
        elif _all_decided(left_bit):
            length = left_bit.size
            llr_matrix[position[0] + 1, position[1] + half : position[1] + span] = g_operation(
                up_llr[:length], up_llr[length:], left_bit
            )
        elif not _all_decided(left_llr):
            length = up_llr.size // 2
            llr_matrix[position[0] + 1, position[1] : position[1] + half] = f_operation(
                up_llr[:length], up_llr[length:]
            )
        else:
            if position[0] == position[2] - 1:
                left_bit_pos = position[1]
                if left_bit_pos in information_pos:
                    val = 0 if left_llr[0] >= 0 else 1
                else:
                    val = frozen_bit
                bit_matrix[position[0] + 1, position[1] : position[1] + half] = val
            else:
                position = [position[0] + 1, position[1], position[2], position[3]]

    return llr_matrix, bit_matrix


def _pm_update_hf(llr_slice, bit_slice):
    pm = 0.0
    for llr_val, bit_val in zip(llr_slice, bit_slice):
        hard = 0 if llr_val >= 0 else 1
        if int(bit_val) != hard:
            pm += abs(llr_val)
    return pm


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.list_size = list_size
        self.crc_length = crc_length
        self.information_pos = _frozen_bits_to_info_pos(frozen_bits)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        if self.list_size == 1 and self.crc_length == 0:
            frozen_bits = np.ones(self.N, dtype=int)
            frozen_bits[self.information_pos] = 0
            return sc_decode(llr_ch, frozen_bits), 0.0

        n, N = self.n, self.N
        info_pos = list(self.information_pos)

        llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
        bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
        llr_matrix[0] = llr_ch

        llr_list = [llr_matrix.copy()]
        bit_list = [bit_matrix.copy()]
        pm_list = [0.0]
        l_now = 1

        for split_loc, split_bit in enumerate(info_pos):
            new_llr, new_bit, new_pm = [], [], []
            for i in range(l_now):
                lm = llr_list[i].copy()
                bm = bit_list[i].copy()
                pm_temp = pm_list[i]

                lm, bm = sc_stepping_decoder(lm, bm, info_pos, 0, split_bit, n, N)

                prev = info_pos[split_loc - 1] if split_loc > 0 else info_pos[-1]
                llr_slice = lm[n, prev + 1 : split_bit + 1]
                bit_slice = bm[n, prev + 1 : split_bit + 1]
                pm_add = _pm_update_hf(llr_slice, bit_slice)

                new_llr.append(lm.copy())
                new_bit.append(bm.copy())
                new_pm.append(pm_temp + pm_add)

                bm_wrong = bm.copy()
                bm_wrong[n, split_bit] = 1 - int(bm_wrong[n, split_bit])
                bit_slice_wrong = bm_wrong[n, prev + 1 : split_bit + 1]
                pm_wrong = pm_temp + _pm_update_hf(llr_slice, bit_slice_wrong)

                new_llr.append(lm.copy())
                new_bit.append(bm_wrong)
                new_pm.append(pm_wrong)

            order = np.argsort(new_pm)[: self.list_size]
            llr_list = [new_llr[i] for i in order]
            bit_list = [new_bit[i] for i in order]
            pm_list = [new_pm[i] for i in order]
            l_now = len(order)

        if info_pos[-1] != N - 1:
            for i in range(l_now):
                lm, bm = sc_stepping_decoder(
                    llr_list[i].copy(), bit_list[i].copy(), info_pos, 0, N - 1, n, N
                )
                prev = info_pos[-1]
                pm_add = _pm_update_hf(lm[n, prev + 1 : N], bm[n, prev + 1 : N])
                llr_list[i] = lm
                bit_list[i] = bm
                pm_list[i] += pm_add

        order = np.argsort(pm_list)
        best_u = None
        best_pm = pm_list[order[0]]

        if self.crc_length > 0:
            for idx in order:
                u_candidate = bit_list[idx][n].astype(int)
                payload = u_candidate[info_pos]
                if crc_check(payload, self.crc_length):
                    best_u = u_candidate
                    best_pm = pm_list[idx]
                    break

        if best_u is None:
            best_u = bit_list[order[0]][n].astype(int)

        return best_u, best_pm
