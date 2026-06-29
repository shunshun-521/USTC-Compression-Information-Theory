"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    sc_decode,
    _frozen_to_info_pos,
    f_operation,
    g_operation,
    _all_filled,
    _leftdown,
    _rightdown,
    _up,
    _get_up_bit,
    _get_right_llr,
    _get_left_llr,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            msb = reg & (1 << (crc_length - 1))
            reg = ((reg << 1) & ((1 << crc_length) - 1)) ^ (poly if msb else 0)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    return np.array_equal(crc_encode(bits[:-crc_length], crc_length), bits)


def _get_up_loc(bit_matrix):
    N = int(bit_matrix.shape[1])
    n = int(math.log2(N))
    detect_array = bit_matrix[n]
    detect = -1
    for i in range(N):
        if detect_array[i] not in (0, 1):
            detect = i - 1
            break
    if detect % 2 == 0:
        loc_row = n - 1
        loc_col = detect
    else:
        loc_row = n - 1
        loc_col = detect - 1
    if detect == -1:
        loc_row = 0
        loc_col = 0
    return [loc_row, loc_col]


def _pm_update(llr_array, bit_array):
    pm = 0.0
    for i in range(llr_array.size):
        if np.sign(llr_array[i]) != np.sign(1 - 2 * bit_array[i]):
            pm += np.abs(llr_array[i])
    return pm


def _sc_stepping_decoder(llr_matrix, bit_matrix, info_set, frozen_val, split_pos):
    N = int(bit_matrix.shape[1])
    n = int(math.log2(N))
    loc = _get_up_loc(bit_matrix)
    position = [loc[0], loc[1], n, N]

    while bit_matrix[n][split_pos] not in (0, 1):
        span = 2 ** (position[2] - position[0])
        half = span // 2
        up_llr = llr_matrix[position[0]][position[1] : position[1] + span]
        up_bit = bit_matrix[position[0]][position[1] : position[1] + span]
        left_llr = llr_matrix[position[0] + 1][position[1] : position[1] + half]
        left_bit = bit_matrix[position[0] + 1][position[1] : position[1] + half]
        right_llr = llr_matrix[position[0] + 1][position[1] + half : position[1] + span]
        right_bit = bit_matrix[position[0] + 1][position[1] + half : position[1] + span]

        if _all_filled(up_bit):
            position = _up(position)
        elif _all_filled(right_bit):
            up_bit_val = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0]][position[1] : position[1] + span] = up_bit_val.copy()
        elif _all_filled(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                if right_bit_pos in info_set:
                    val = 0 if right_llr[0] >= 0 else 1
                else:
                    val = frozen_val
                bit_matrix[position[0] + 1][position[1] + half : position[1] + span] = val
            else:
                position = _rightdown(position)
        elif _all_filled(left_bit):
            right_llr_val = _get_right_llr(left_bit, up_llr)
            llr_matrix[position[0] + 1][position[1] + half : position[1] + span] = (
                right_llr_val
            )
        elif not _all_filled(left_llr):
            left_llr_val = _get_left_llr(up_llr)
            llr_matrix[position[0] + 1][position[1] : position[1] + half] = left_llr_val
        else:
            if position[0] == position[2] - 1:
                left_bit_pos = position[1]
                if left_bit_pos in info_set:
                    val = 0 if left_llr[0] >= 0 else 1
                else:
                    val = frozen_val
                bit_matrix[position[0] + 1][position[1] : position[1] + half] = val
            else:
                position = _leftdown(position)

    return llr_matrix, bit_matrix


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits)
        if self.frozen_bits.dtype != bool:
            self.frozen_bits = self.frozen_bits.astype(bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = _frozen_to_info_pos(self.frozen_bits)
        self.info_set = set(int(i) for i in self.info_indices)

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        N, n, L = self.N, self.n, self.list_size
        split_pos = list(self.info_indices)
        frozen_val = 0

        llr_matrix = np.ones((n + 1, N))
        llr_matrix[llr_matrix == 1] = np.nan
        bit_matrix = llr_matrix.copy()
        llr_matrix[0] = llr_ch

        llr_list = [llr_matrix.copy()]
        bit_list = [bit_matrix.copy()]
        pm_list = [0.0]
        split_loc = 0
        l_now = 1

        while split_loc < len(split_pos):
            new_llr, new_bit, new_pm = [], [], []
            for i in range(l_now):
                lm, bm = llr_list[i].copy(), bit_list[i].copy()
                lm, bm = _sc_stepping_decoder(
                    lm, bm, self.info_set, frozen_val, split_pos[split_loc]
                )
                prev = split_pos[split_loc - 1] + 1 if split_loc > 0 else 0
                cur = split_pos[split_loc] + 1
                pm_add = _pm_update(lm[n][prev:cur], bm[n][prev:cur])

                new_llr.append(lm)
                new_bit.append(bm)
                new_pm.append(pm_list[i] + pm_add)

                bm_wrong = bm.copy()
                bm_wrong[n][split_pos[split_loc]] = 1 - bm_wrong[n][split_pos[split_loc]]
                pm_wrong = pm_list[i] + _pm_update(
                    lm[n][prev:cur], bm_wrong[n][prev:cur]
                )
                new_llr.append(lm.copy())
                new_bit.append(bm_wrong)
                new_pm.append(pm_wrong)

            order = np.argsort(new_pm)[:L]
            llr_list = [new_llr[i] for i in order]
            bit_list = [new_bit[i] for i in order]
            pm_list = [new_pm[i] for i in order]
            l_now = len(pm_list)
            split_loc += 1

        if split_pos and split_pos[-1] != N - 1:
            for i in range(l_now):
                prev = split_pos[-1] + 1
                lm, bm = llr_list[i].copy(), bit_list[i].copy()
                lm, bm = _sc_stepping_decoder(lm, bm, self.info_set, frozen_val, N - 1)
                pm_list[i] += _pm_update(lm[n][prev:N], bm[n][prev:N])
                llr_list[i], bit_list[i] = lm, bm

        order = np.argsort(pm_list)
        best_u, best_pm = None, np.inf
        for idx in order:
            u_hat = bit_list[idx][n].astype(int)
            if self.crc_length > 0:
                if crc_check(u_hat[self.info_indices], self.crc_length):
                    return u_hat, pm_list[idx]
            elif pm_list[idx] < best_pm:
                best_u, best_pm = u_hat, pm_list[idx]

        if best_u is None:
            best_u = bit_list[order[0]][n].astype(int)
            best_pm = pm_list[order[0]]
        return best_u, best_pm
