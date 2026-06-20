"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import f_operation, g_operation, sc_decode


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_divide(info_bits, poly, crc_length):
    bits = [int(b) for b in info_bits]
    poly_bits = [(poly >> i) & 1 for i in range(crc_length, -1, -1)]
    poly_len = len(poly_bits)
    for _ in range(crc_length):
        bits.append(0)
    for i in range(len(bits) - crc_length):
        if bits[i] == 1:
            for j in range(poly_len):
                bits[i + j] ^= poly_bits[j]
    return np.array(bits[-crc_length:], dtype=int)


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    crc_bits = _crc_divide(info_bits, poly, crc_length)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    if crc_length == 0:
        return True
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_divide(bits, poly, crc_length)
    return np.all(remainder == 0)


def _frozen_mask_to_bool(frozen_bits):
    fb = np.asarray(frozen_bits)
    return fb.astype(bool) if fb.dtype != bool else fb


def _all_known(bits):
    return not np.any(np.isnan(bits))


def _up(pos):
    pos[0] -= 1
    block = 1 << (pos[2] - pos[0] + 1)
    pos[1] = int(np.floor(pos[1] / block) * block)


def _leftdown(pos):
    pos[0] += 1


def _rightdown(pos):
    pos[0] += 1
    pos[1] += 1 << (pos[2] - 1 - pos[0] + 1)


def _get_up_bit(left_bit, right_bit):
    length = len(left_bit)
    out = np.empty(2 * length, dtype=np.float64)
    out[:length] = (left_bit + right_bit) % 2
    out[length:] = right_bit
    return out


def _get_up_loc(bit_matrix, n, N):
    for i in range(N):
        v = bit_matrix[n, i]
        if np.isnan(v) or v not in (0, 1):
            detect = i - 1
            break
    else:
        detect = N - 1
    if detect < 0:
        return [0, 0]
    if detect % 2 == 0:
        return [n - 1, detect]
    return [n - 1, max(detect - 1, 0)]


def _sc_step_to_bit(llr_matrix, bit_matrix, frozen_bits, stop_phi, n, N, max_iter=10000):
    """树遍历推进至 bit_matrix[n, stop_phi] 判决完成。"""
    loc = _get_up_loc(bit_matrix, n, N)
    position = [loc[0], loc[1], n, N]
    iters = 0

    while np.isnan(bit_matrix[n, stop_phi]) and iters < max_iter:
        iters += 1
        row, col, depth, _ = position
        span = 1 << (depth - row)
        up_llr = llr_matrix[row, col : col + span]
        up_bit = bit_matrix[row, col : col + span]
        half = span // 2
        left_llr = llr_matrix[row + 1, col : col + half]
        left_bit = bit_matrix[row + 1, col : col + half]
        right_llr = llr_matrix[row + 1, col + half : col + span]
        right_bit = bit_matrix[row + 1, col + half : col + span]

        if _all_known(up_bit):
            _up(position)
            continue
        if _all_known(right_bit):
            bit_matrix[row, col : col + span] = _get_up_bit(left_bit, right_bit)
            continue
        if _all_known(right_llr):
            if row == depth - 1:
                right_pos = col + half
                if frozen_bits[right_pos]:
                    bit_matrix[row + 1, right_pos] = 0
                else:
                    bit_matrix[row + 1, right_pos] = 0 if right_llr[0] >= 0 else 1
            else:
                _rightdown(position)
            continue
        if _all_known(left_bit):
            llr_matrix[row + 1, col + half : col + span] = g_operation(
                up_llr[:half], up_llr[half:], left_bit
            )
            continue
        if not _all_known(left_llr):
            llr_matrix[row + 1, col : col + half] = f_operation(up_llr[:half], up_llr[half:])
            continue
        if row == depth - 1:
            left_pos = col
            if frozen_bits[left_pos]:
                bit_matrix[row + 1, left_pos] = 0
            else:
                bit_matrix[row + 1, left_pos] = 0 if left_llr[0] >= 0 else 1
        else:
            _leftdown(position)

    return llr_matrix, bit_matrix


def _pm_hf(llr_slice, bit_slice):
    pm = 0.0
    for lv, bv in zip(llr_slice, bit_slice):
        expected = 0 if lv >= 0 else 1
        if bv != expected:
            pm += abs(lv)
    return pm


class SCLDecoder:
    """SCL 译码器（信息位分裂 + CRC 辅助）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = _frozen_mask_to_bool(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr_list = [np.full((n + 1, N), np.nan)]
        bit_list = [np.full((n + 1, N), np.nan)]
        llr_list[0][0, :] = llr_ch
        pm_list = [0.0]

        split_positions = list(self.info_indices)
        if split_positions[-1] != N - 1:
            split_positions.append(N - 1)

        for si, split_phi in enumerate(split_positions):
            prev_phi = split_positions[si - 1] if si > 0 else -1
            new_llr, new_bit, new_pm = [], [], []

            for idx in range(len(llr_list)):
                lm = llr_list[idx].copy()
                bm = bit_list[idx].copy()
                pm0 = pm_list[idx]

                lm, bm = _sc_step_to_bit(lm, bm, self.frozen_bits, split_phi, n, N)

                if self.frozen_bits[split_phi]:
                    bm[n, split_phi] = 0
                    seg_llr = lm[n, prev_phi + 1 : split_phi + 1]
                    seg_bit = bm[n, prev_phi + 1 : split_phi + 1].astype(int)
                    new_llr.append(lm)
                    new_bit.append(bm)
                    new_pm.append(pm0 + _pm_hf(seg_llr, seg_bit))
                else:
                    for u_bit in (0, 1):
                        lm_c = lm.copy()
                        bm_c = bm.copy()
                        bm_c[n, split_phi] = u_bit
                        seg_llr = lm_c[n, prev_phi + 1 : split_phi + 1]
                        seg_bit = bm_c[n, prev_phi + 1 : split_phi + 1].astype(int)
                        new_llr.append(lm_c)
                        new_bit.append(bm_c)
                        new_pm.append(pm0 + _pm_hf(seg_llr, seg_bit))

            order = np.argsort(new_pm)[: self.list_size]
            llr_list = [new_llr[i] for i in order]
            bit_list = [new_bit[i] for i in order]
            pm_list = [new_pm[i] for i in order]

        best = None
        best_pm = float("inf")
        for i in np.argsort(pm_list):
            u_hat = bit_list[i][n].astype(int)
            if self.crc_length > 0:
                if crc_check(u_hat[self.info_indices], self.crc_length):
                    return u_hat, pm_list[i]
            else:
                return u_hat, pm_list[i]

        i = int(np.argmin(pm_list))
        return bit_list[i][n].astype(int), pm_list[i]
