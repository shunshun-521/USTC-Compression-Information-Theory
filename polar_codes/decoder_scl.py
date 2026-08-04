"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math
from decoder_sc import _prepare_llr, _info_positions, _sc_tree_decode, f_operation, g_operation


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg = ((reg << 1) | int(bit)) & ((1 << crc_length) - 1)
        if reg & (1 << (crc_length - 1)):
            reg ^= poly
    for _ in range(crc_length):
        reg = (reg << 1) & ((1 << crc_length) - 1)
        if reg & (1 << (crc_length - 1)):
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)])
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 的 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


def _all_computed(x):
    return not np.any(np.isnan(x))


def _pm_update(llrs, bits):
    pm = 0.0
    for llr, u in zip(llrs, bits):
        hard = 0 if llr >= 0 else 1
        if int(u) != hard:
            pm += abs(llr)
    return pm


def _get_up_loc(bit_matrix):
    """定位树遍历当前位置（参考实现）"""
    N = bit_matrix.shape[1]
    n = int(math.log2(N))
    detect_array = bit_matrix[n]
    detect = -1
    for i in range(N):
        if detect_array[i] not in (0, 1):
            detect = i - 1
            break
    if detect == -1:
        return [0, 0, n, N]
    if detect % 2 == 0:
        loc_row = n - 1
        loc_col = detect
    else:
        loc_row = n - 1
        loc_col = detect - 1
    return [loc_row, loc_col, n, N]


def _sc_step_to_phi(llr_matrix, bit_matrix, info_pos, stop_phi, frozen_val=0):
    """SC 译码至 stop_phi（含判决）"""
    N = bit_matrix.shape[1]
    n = int(math.log2(N))
    info_set = set(info_pos)
    position = _get_up_loc(bit_matrix)

    def leftdown(p):
        return [p[0] + 1, p[1], p[2], p[3]]

    def rightdown(p):
        return [p[0] + 1, p[1] + 2 ** (p[2] - 1 - p[0]), p[2], p[3]]

    def up(p):
        return [p[0] - 1, int(np.floor(p[1] / (2 ** (p[2] - p[0] + 1))) * (2 ** (p[2] - p[0] + 1))), p[2], p[3]]

    while np.isnan(bit_matrix[n, stop_phi]):
        span = 2 ** (position[2] - position[0])
        s, e = position[1], position[1] + span
        half = span // 2
        up_llr = llr_matrix[position[0]][s:e]
        up_bit = bit_matrix[position[0]][s:e]
        left_llr = llr_matrix[position[0] + 1][s:s + half]
        left_bit = bit_matrix[position[0] + 1][s:s + half]
        right_llr = llr_matrix[position[0] + 1][s + half:e]
        right_bit = bit_matrix[position[0] + 1][s + half:e]

        if _all_computed(up_bit):
            position = up(position)
        elif _all_computed(right_bit):
            length = len(left_bit)
            temp = np.array([(left_bit + right_bit) % 2, right_bit])
            temp.resize((1, 2 * length))
            bit_matrix[position[0]][s:e] = temp.flatten()
        elif _all_computed(right_llr):
            if position[0] == position[2] - 1:
                for i in range(half):
                    pos = s + half + i
                    bit_matrix[position[0] + 1][pos] = (0 if right_llr[i] > 0 else 1) if pos in info_set else frozen_val
            else:
                position = rightdown(position)
        elif _all_computed(left_bit):
            llr_matrix[position[0] + 1][s + half:e] = g_operation(up_llr[:half], up_llr[half:], left_bit)
        elif not _all_computed(left_llr):
            llr_matrix[position[0] + 1][s:s + half] = f_operation(up_llr[:half], up_llr[half:])
        else:
            if position[0] == position[2] - 1:
                for i in range(half):
                    pos = s + i
                    bit_matrix[position[0] + 1][pos] = (0 if left_llr[i] >= 0 else 1) if pos in info_set else frozen_val
            else:
                position = leftdown(position)

    return llr_matrix, bit_matrix


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = _info_positions(self.frozen_bits)

    def decode(self, llr_ch):
        """主译码函数"""
        N, n = self.N, self.n
        llr = _prepare_llr(llr_ch, N)
        info_pos = list(self.info_indices)

        llr_list = [np.full((n + 1, N), np.nan)]
        bit_list = [np.full((n + 1, N), np.nan)]
        llr_list[0][0] = llr
        pm_list = [0.0]

        split_positions = list(self.info_indices)
        prev_phi = -1

        for split_phi in split_positions:
            new_llr, new_bit, new_pm = [], [], []
            for llr_m, bit_m, pm in zip(llr_list, bit_list, pm_list):
                llr_m, bit_m = _sc_step_to_phi(llr_m.copy(), bit_m.copy(), info_pos, split_phi)
                seg_llr = llr_m[n, prev_phi + 1:split_phi + 1]
                seg_bit = bit_m[n, prev_phi + 1:split_phi + 1]
                pm_add = _pm_update(seg_llr, seg_bit)

                new_llr.append(llr_m)
                new_bit.append(bit_m)
                new_pm.append(pm + pm_add)

                bm1 = bit_m.copy()
                bm1[n, split_phi] = 1 - bm1[n, split_phi]
                wrong_pm = pm + _pm_update(seg_llr, bm1[n, prev_phi + 1:split_phi + 1])
                new_llr.append(llr_m.copy())
                new_bit.append(bm1)
                new_pm.append(wrong_pm)

            order = np.argsort(new_pm)[:self.list_size]
            llr_list = [new_llr[i] for i in order]
            bit_list = [new_bit[i] for i in order]
            pm_list = [new_pm[i] for i in order]
            prev_phi = split_phi

        if prev_phi < N - 1:
            for i in range(len(llr_list)):
                llr_list[i], bit_list[i] = _sc_step_to_phi(llr_list[i], bit_list[i], info_pos, N - 1)

        candidates = []
        for bit_m, pm in zip(bit_list, pm_list):
            u_hat = np.nan_to_num(bit_m[n], nan=0).astype(int)
            candidates.append((u_hat, pm))

        if self.crc_length > 0:
            valid = [(u, p) for u, p in candidates if crc_check(u[self.info_indices], self.crc_length)]
            if valid:
                best = min(valid, key=lambda x: x[1])
                return best[0], best[1]

        best = min(candidates, key=lambda x: x[1])
        return best[0], best[1]
