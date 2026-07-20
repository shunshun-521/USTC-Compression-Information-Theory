"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy

import numpy as np

from decoder_sc import f_operation, g_operation


def _crc_poly(crc_length):
    if crc_length == 8:
        return [8, 2, 1, 0]
    if crc_length == 16:
        return [16, 15, 2, 0]
    raise ValueError(f"Unsupported CRC length: {crc_length}")


def _crc_remainder(info_bits, crc_length):
    loc = _crc_poly(crc_length)
    p = [0] * (crc_length + 1)
    for i in loc:
        p[i] = 1
    p = p[::-1]

    bits = list(info_bits) + [0] * crc_length
    times = len(info_bits)
    for i in range(times):
        if bits[i] == 1:
            for j in range(crc_length + 1):
                bits[i + j] ^= p[j]
    return bits[-crc_length:]


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    remainder = _crc_remainder(info_bits.tolist(), crc_length)
    return np.concatenate([info_bits, remainder])


def crc_check(bits, crc_length=8):
    """检验 bits 是否通过 CRC 校验"""
    bits = np.asarray(bits, dtype=int).ravel()
    if len(bits) < crc_length:
        return False
    info = bits[:-crc_length]
    expected = _crc_remainder(info.tolist(), crc_length)
    return np.array_equal(bits[-crc_length:], expected)


def _all_filled(arr):
    return not np.any(np.isnan(arr))


def _get_up_loc(bit_matrix, N, n):
    detect_array = bit_matrix[n]
    detect = -1
    for i in range(N):
        if np.isnan(detect_array[i]):
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
    return loc_row, loc_col


def _pm_update(llr_array, bit_array):
    pm = 0.0
    for llr, bit in zip(llr_array, bit_array):
        hard = 0 if llr >= 0 else 1
        if hard != bit:
            pm += abs(llr)
    return pm


def _sc_step_to(llr_matrix, bit_matrix, information_pos, frozen_bit, split_pos, N, n):
    """SC 译码推进到 split_pos 判决完成"""
    loc_row, loc_col = _get_up_loc(bit_matrix, N, n)
    position = [loc_row, loc_col, n, N]

    while np.isnan(bit_matrix[n, split_pos]):
        block_len = 2 ** (position[2] - position[0])
        up_llr = llr_matrix[position[0], position[1]:position[1] + block_len]
        up_bit = bit_matrix[position[0], position[1]:position[1] + block_len]
        half = block_len // 2
        left_llr = llr_matrix[position[0] + 1, position[1]:position[1] + half]
        left_bit = bit_matrix[position[0] + 1, position[1]:position[1] + half]
        right_llr = llr_matrix[position[0] + 1, position[1] + half:position[1] + block_len]
        right_bit = bit_matrix[position[0] + 1, position[1] + half:position[1] + block_len]

        if _all_filled(up_bit):
            position[0] -= 1
            position[1] = int(np.floor(position[1] / (2 ** (position[2] - position[0]))) *
                             (2 ** (position[2] - position[0])))
        elif _all_filled(right_bit):
            new_up = np.concatenate([(left_bit + right_bit) % 2, right_bit])
            bit_matrix[position[0], position[1]:position[1] + block_len] = new_up
        elif _all_filled(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + half
                if right_bit_pos in information_pos:
                    bit_val = 0 if right_llr[0] >= 0 else 1
                else:
                    bit_val = frozen_bit
                bit_matrix[position[0] + 1, right_bit_pos] = bit_val
            else:
                position[0] += 1
                position[1] += half
        elif _all_filled(left_bit):
            right_llr_new = g_operation(up_llr[:half], up_llr[half:], left_bit)
            llr_matrix[position[0] + 1, position[1] + half:position[1] + block_len] = right_llr_new
        elif not _all_filled(left_llr):
            left_llr_new = f_operation(up_llr[:half], up_llr[half:])
            llr_matrix[position[0] + 1, position[1]:position[1] + half] = left_llr_new
        else:
            if position[0] == position[2] - 1:
                left_bit_pos = position[1]
                if left_bit_pos in information_pos:
                    bit_val = 0 if left_llr[0] >= 0 else 1
                else:
                    bit_val = frozen_bit
                bit_matrix[position[0] + 1, left_bit_pos] = bit_val
            else:
                position[0] += 1

    return llr_matrix, bit_matrix


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.information_pos = np.where(~self.frozen_bits)[0]
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_bit = 0

    def decode(self, llr_ch):
        """主译码函数"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        info_set = set(self.information_pos.tolist())

        llr0 = np.full((n + 1, N), np.nan, dtype=np.float64)
        bit0 = np.full((n + 1, N), np.nan, dtype=np.float64)
        llr0[0] = llr_ch

        llr_list = [llr0]
        bit_list = [bit0]
        pm_list = [0.0]

        split_positions = [p for p in self.information_pos if p not in info_set or True]
        split_positions = list(self.information_pos)

        prev_pos = -1
        for split_pos in split_positions:
            new_llr, new_bit, new_pm = [], [], []
            for llr_m, bit_m, pm in zip(llr_list, bit_list, pm_list):
                llr_m, bit_m = _sc_step_to(
                    llr_m.copy(), bit_m.copy(), info_set, self.frozen_bit, split_pos, N, n
                )
                llr_seg = llr_m[n, prev_pos + 1:split_pos + 1]
                bit_seg = bit_m[n, prev_pos + 1:split_pos + 1]
                pm_add = _pm_update(llr_seg, bit_seg)

                new_llr.append(llr_m)
                new_bit.append(bit_m)
                new_pm.append(pm + pm_add)

                bit_wrong = bit_m.copy()
                bit_wrong[n, split_pos] = 1 - bit_wrong[n, split_pos]
                bit_seg_w = bit_wrong[n, prev_pos + 1:split_pos + 1]
                pm_wrong = pm + _pm_update(llr_seg, bit_seg_w)

                new_llr.append(llr_m.copy())
                new_bit.append(bit_wrong)
                new_pm.append(pm_wrong)

            order = np.argsort(new_pm)
            keep = order[:self.list_size]
            llr_list = [new_llr[i] for i in keep]
            bit_list = [new_bit[i] for i in keep]
            pm_list = [new_pm[i] for i in keep]
            prev_pos = split_pos

        if split_positions[-1] != N - 1:
            final_llr, final_bit, final_pm = [], [], []
            for llr_m, bit_m, pm in zip(llr_list, bit_list, pm_list):
                llr_m, bit_m = _sc_step_to(
                    llr_m.copy(), bit_m.copy(), info_set, self.frozen_bit, N - 1, N, n
                )
                llr_seg = llr_m[n, prev_pos + 1:N]
                bit_seg = bit_m[n, prev_pos + 1:N]
                final_llr.append(llr_m)
                final_bit.append(bit_m)
                final_pm.append(pm + _pm_update(llr_seg, bit_seg))
            order = np.argsort(final_pm)
            keep = order[:self.list_size]
            bit_list = [final_bit[i] for i in keep]
            pm_list = [final_pm[i] for i in keep]

        order = np.argsort(pm_list)
        if self.crc_length > 0:
            for idx in order:
                u_hat = bit_list[idx][n].astype(int)
                info_bits = u_hat[self.information_pos]
                if crc_check(info_bits, self.crc_length):
                    return u_hat, pm_list[idx]
            idx = order[0]
            return bit_list[idx][n].astype(int), pm_list[idx]

        idx = order[0]
        return bit_list[idx][n].astype(int), pm_list[idx]
