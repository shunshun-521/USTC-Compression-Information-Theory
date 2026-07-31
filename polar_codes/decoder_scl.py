"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    _combine_left_llr,
    _combine_right_llr,
    sc_decode,
)

# ==================== CRC 工具 ====================

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg = (reg << 1) | int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    return reg & ((1 << crc_length) - 1)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


# ==================== SCL 辅助 ====================


def _all_filled(arr):
    return not np.any(np.isnan(arr))


def _leftdown(pos):
    return [pos[0] + 1, pos[1], pos[2], pos[3]]


def _rightdown(pos):
    return [pos[0] + 1, pos[1] + 2 ** (pos[2] - 1 - pos[0]), pos[2], pos[3]]


def _up(pos):
    p0 = pos[0] - 1
    p1 = int(np.floor(pos[1] / (2 ** (pos[2] - pos[0] + 1))) * (2 ** (pos[2] - pos[0] + 1)))
    return [p0, p1, pos[2], pos[3]]


def _get_up_loc(bit_matrix):
    N = bit_matrix.shape[1]
    n = int(math.log2(N))
    detect = -1
    for i in range(N):
        if bit_matrix[n, i] != 0 and bit_matrix[n, i] != 1:
            detect = i - 1
            break
    if detect == -1:
        return [0, 0]
    if detect % 2 == 0:
        return [n - 1, detect]
    return [n - 1, detect - 1]


def _pm_penalty(llr, bit):
    hard = 0 if llr >= 0 else 1
    return 0.0 if bit == hard else abs(llr)


def _pm_update(llr_array, bit_array):
    return sum(_pm_penalty(llr_array[i], int(bit_array[i])) for i in range(len(llr_array)))


def _sc_step_to_bit(llr_matrix, bit_matrix, frozen_bits, target_bit):
    """树形 SC 逐步译码，直到完成 target_bit 的判决。"""
    N = bit_matrix.shape[1]
    n = int(math.log2(N))
    loc = _get_up_loc(bit_matrix)
    position = [loc[0], loc[1], n, N]

    while bit_matrix[n, target_bit] != 0 and bit_matrix[n, target_bit] != 1:
        span = 2 ** (position[2] - position[0])
        start = position[1]
        up_llr = llr_matrix[position[0]][start:start + span]
        left_llr = llr_matrix[position[0] + 1][start:start + span // 2]
        left_bit = bit_matrix[position[0] + 1][start:start + span // 2]
        right_llr = llr_matrix[position[0] + 1][start + span // 2:start + span]
        right_bit = bit_matrix[position[0] + 1][start + span // 2:start + span]

        if _all_filled(bit_matrix[position[0]][start:start + span]):
            position = _up(position)
        elif _all_filled(right_bit):
            combined = np.array([(left_bit + right_bit) % 2, right_bit]).reshape(1, -1)
            bit_matrix[position[0]][start:start + span] = combined
        elif _all_filled(right_llr):
            if position[0] == position[2] - 1:
                idx = position[1] + 1
                val = 0 if frozen_bits[idx] or right_llr[0] >= 0 else 1
                bit_matrix[position[0] + 1][start + span // 2:start + span] = val
            else:
                position = _rightdown(position)
        elif _all_filled(left_bit):
            llr_matrix[position[0] + 1][start + span // 2:start + span] = _combine_right_llr(up_llr, left_bit)
        elif not _all_filled(left_llr):
            llr_matrix[position[0] + 1][start:start + span // 2] = _combine_left_llr(up_llr)
        else:
            if position[0] == position[2] - 1:
                idx = position[1]
                val = 0 if frozen_bits[idx] or left_llr[0] >= 0 else 1
                bit_matrix[position[0] + 1][start:start + span // 2] = val
            else:
                position = _leftdown(position)

    return llr_matrix, bit_matrix


# ==================== SCL 译码器 ====================


class SCLDecoder:
    """SCL 译码器（树形 SC 扩展，支持 CRC 辅助）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)。"""
        N = self.N
        n = self.n

        if self.list_size == 1:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr0 = np.full((n + 1, N), np.nan, dtype=np.float64)
        bit0 = np.full((n + 1, N), np.nan, dtype=np.float64)
        llr0[0] = llr_ch

        llr_list = [llr0]
        bit_list = [bit0]
        pm_list = [0.0]

        prev_split = -1
        split_positions = list(self.info_indices)

        for split_pos in split_positions:
            new_llr_list = []
            new_bit_list = []
            new_pm_list = []

            for llr_m, bit_m, pm in zip(llr_list, bit_list, pm_list):
                llr_m, bit_m = _sc_step_to_bit(
                    llr_m.copy(), bit_m.copy(), self.frozen_bits, split_pos
                )
                llr_seg = llr_m[n, prev_split + 1:split_pos + 1]
                bit_seg = bit_m[n, prev_split + 1:split_pos + 1]

                for bit_val in (0, 1):
                    if self.frozen_bits[split_pos] and bit_val != 0:
                        continue
                    bm = bit_m.copy()
                    if not self.frozen_bits[split_pos]:
                        bm[n, split_pos] = bit_val
                    penalty = _pm_update(llr_seg, bm[n, prev_split + 1:split_pos + 1])
                    new_llr_list.append(llr_m.copy())
                    new_bit_list.append(bm)
                    new_pm_list.append(pm + penalty)

            order = np.argsort(new_pm_list)
            keep = order[: self.list_size]
            llr_list = [new_llr_list[i] for i in keep]
            bit_list = [new_bit_list[i] for i in keep]
            pm_list = [new_pm_list[i] for i in keep]
            prev_split = split_pos

        # 完成剩余冻结位译码
        final_llr = []
        final_bit = []
        final_pm = []
        for llr_m, bit_m, pm in zip(llr_list, bit_list, pm_list):
            if split_positions:
                last = split_positions[-1]
                if last < N - 1:
                    llr_m, bit_m = _sc_step_to_bit(
                        llr_m.copy(), bit_m.copy(), self.frozen_bits, N - 1
                    )
                    seg_llr = llr_m[n, last + 1:N]
                    seg_bit = bit_m[n, last + 1:N]
                    pm += _pm_update(seg_llr, seg_bit)
            final_llr.append(llr_m)
            final_bit.append(bit_m)
            final_pm.append(pm)

        order = np.argsort(final_pm)
        best_crc = None
        if self.crc_length > 0:
            for idx in order:
                u_cand = final_bit[idx][n].astype(int)
                info_bits = u_cand[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    best_crc = (u_cand, final_pm[idx])
                    break

        if best_crc is not None:
            return best_crc[0], best_crc[1]

        best_idx = order[0]
        return final_bit[best_idx][n].astype(int), final_pm[best_idx]
