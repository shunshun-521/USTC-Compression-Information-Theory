"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    f_operation,
    g_operation,
    _all_decided,
    _decide_bit,
    _get_up_bit,
    _info_indices,
    _leftdown,
    _rightdown,
    _up,
)
from encoder import prepare_channel_llr


CRC8_GEN = [1, 0, 0, 0, 0, 0, 1, 1, 1]       # x^8 + x^2 + x + 1
CRC16_GEN = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  # CRC-16-IBM


def _crc_generator(crc_length):
    if crc_length == 8:
        return CRC8_GEN
    if crc_length == 16:
        return CRC16_GEN
    raise ValueError(f"Unsupported CRC length: {crc_length}")


def _crc_divide(data_bits, generator):
    """二进制长除法求 CRC 余数"""
    msg = list(np.asarray(data_bits, dtype=int))
    r = len(generator) - 1
    msg.extend([0] * r)
    for i in range(len(data_bits)):
        if msg[i]:
            for j, g in enumerate(generator):
                if g:
                    msg[i + j] ^= 1
    return np.array(msg[len(data_bits):], dtype=int)


def _crc_verify(bits, generator):
    msg = list(np.asarray(bits, dtype=int))
    for i in range(len(bits) - (len(generator) - 1)):
        if msg[i]:
            for j, g in enumerate(generator):
                if g:
                    msg[i + j] ^= 1
    return all(v == 0 for v in msg[-(len(generator) - 1):])


def crc_encode(info_bits, crc_length=8):
    """CRC-8 (0x07) 或 CRC-16 (0x8005)"""
    info_bits = np.asarray(info_bits, dtype=int)
    gen = _crc_generator(crc_length)
    crc_bits = _crc_divide(info_bits, gen)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    gen = _crc_generator(crc_length)
    return _crc_verify(bits, gen)


def _get_up_loc(bit_matrix, n, N):
    detect = -1
    for i in range(N):
        if np.isnan(bit_matrix[n, i]):
            detect = i - 1
            break
    if detect == -1:
        return [0, 0]
    if detect % 2 == 0:
        return [n - 1, detect]
    return [n - 1, detect - 1]


def _sc_advance(llr_matrix, bit_matrix, info_set, frozen_val, stop_phi):
    """推进 SC 状态直至 bit_matrix[n, stop_phi] 被判决（或已完成）"""
    N = bit_matrix.shape[1]
    n = int(math.log2(N))

    if not np.isnan(bit_matrix[n, stop_phi]):
        return llr_matrix, bit_matrix

    loc = _get_up_loc(bit_matrix, n, N)
    position = [loc[0], loc[1], n, N]

    while np.isnan(bit_matrix[n, stop_phi]):
        span = 2 ** (position[2] - position[0])
        p0, p1 = position[0], position[1]

        up_llr = llr_matrix[p0][p1:p1 + span]
        up_bit = bit_matrix[p0][p1:p1 + span]
        left_llr = llr_matrix[p0 + 1][p1:p1 + span // 2]
        left_bit = bit_matrix[p0 + 1][p1:p1 + span // 2]
        right_llr = llr_matrix[p0 + 1][p1 + span // 2:p1 + span]
        right_bit = bit_matrix[p0 + 1][p1 + span // 2:p1 + span]

        if _all_decided(up_bit):
            position = _up(position)
        elif _all_decided(right_bit):
            bit_matrix[p0][p1:p1 + span] = _get_up_bit(left_bit, right_bit)
        elif _all_decided(right_llr):
            if position[0] == position[2] - 1:
                right_pos = p1 + 1
                val = _decide_bit(right_llr[0], right_pos, info_set, frozen_val)
                bit_matrix[p0 + 1][p1 + span // 2:p1 + span] = val
            else:
                position = _rightdown(position)
        elif _all_decided(left_bit):
            half = len(up_llr) // 2
            llr_matrix[p0 + 1][p1 + span // 2:p1 + span] = np.array([
                g_operation(up_llr[i], up_llr[i + half], left_bit[i])
                for i in range(half)
            ])
        elif not _all_decided(left_llr):
            llr_matrix[p0 + 1][p1:p1 + span // 2] = f_operation(
                up_llr[: len(up_llr) // 2],
                up_llr[len(up_llr) // 2:],
            )
        else:
            if position[0] == position[2] - 1:
                left_pos = p1
                val = _decide_bit(left_llr[0], left_pos, info_set, frozen_val)
                bit_matrix[p0 + 1][p1:p1 + span // 2] = val
            else:
                position = _leftdown(position)

    return llr_matrix, bit_matrix


def _pm_segment(llr_matrix, bit_matrix, start_phi, end_phi, n):
    pm = 0.0
    for phi in range(start_phi, end_phi + 1):
        llr = llr_matrix[n, phi]
        bit = bit_matrix[n, phi]
        if np.isnan(llr) or np.isnan(bit):
            continue
        hard = 0 if llr >= 0 else 1
        if int(bit) != hard:
            pm += abs(llr)
    return pm


class SCLDecoder:
    """SCL 译码器（Lazy Copy：分裂时复制 LLR/比特矩阵）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.info_indices = _info_indices(self.frozen_bits)
        self.info_set = set(self.info_indices.tolist())
        self.list_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        y_llr = prepare_channel_llr(llr_ch)
        N, n = self.N, self.n
        frozen_val = 0

        def new_state():
            llr_m = np.full((n + 1, N), np.nan, dtype=np.float64)
            bit_m = np.full((n + 1, N), np.nan, dtype=np.float64)
            llr_m[0] = y_llr
            return llr_m, bit_m

        paths = [(new_state(), 0.0)]
        prev_phi = -1

        for phi in range(N):
            expanded = []
            for (llr_m, bit_m), base_pm in paths:
                llr_m, bit_m = _sc_advance(
                    llr_m.copy(), bit_m.copy(), self.info_set, frozen_val, phi
                )
                leaf_llr = llr_m[n, phi]
                if np.isnan(leaf_llr):
                    leaf_llr = llr_m[0, 0]

                if self.frozen_bits[phi]:
                    bit_m[n, phi] = 0
                    seg_pm = _pm_segment(llr_m, bit_m, max(0, prev_phi + 1), phi, n)
                    expanded.append(((llr_m, bit_m), base_pm + seg_pm))
                else:
                    for bit_val in (0, 1):
                        lm = llr_m.copy()
                        bm = bit_m.copy()
                        bm[n, phi] = bit_val
                        seg_pm = _pm_segment(lm, bm, max(0, prev_phi + 1), phi, n)
                        expanded.append(((lm, bm), base_pm + seg_pm))

            expanded.sort(key=lambda x: x[1])
            paths = expanded[: self.list_size]
            prev_phi = phi

        # 完成剩余 SC 步骤
        finalized = []
        for (llr_m, bit_m), pm in paths:
            for p in range(N):
                if np.isnan(bit_m[n, p]):
                    llr_m, bit_m = _sc_advance(
                        llr_m.copy(), bit_m.copy(), self.info_set, frozen_val, p
                    )
            finalized.append((bit_m[n].astype(int), pm))

        best_u = None
        best_pm = float("inf")
        for u_hat, pm in sorted(finalized, key=lambda x: x[1]):
            if self.crc_length > 0:
                payload = u_hat[self.info_indices]
                if crc_check(payload, self.crc_length):
                    if pm < best_pm:
                        best_pm = pm
                        best_u = u_hat
            elif pm < best_pm:
                best_pm = pm
                best_u = u_hat

        if best_u is None:
            best_u, best_pm = sorted(finalized, key=lambda x: x[1])[0]

        return best_u, best_pm
