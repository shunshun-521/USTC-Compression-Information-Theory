"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    _sc_decode_core,
    _all_filled,
    _left_llr,
    _right_llr,
    _up_bit,
    _leftdown,
    _rightdown,
    _up,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


def _pm_penalty(llr, u):
    hard = 0 if llr >= 0 else 1
    return 0.0 if u == hard else abs(llr)


def _get_up_loc(bit_mat, n):
    detect = -1
    for i in range(bit_mat.shape[1]):
        v = bit_mat[n, i]
        if v != 0 and v != 1 and not (v == 0.0 or v == 1.0):
            detect = i - 1
            break
    if detect == -1:
        return 0, 0
    if detect % 2 == 0:
        return n - 1, detect
    return n - 1, detect - 1


def _sc_step(llr_mat, bit_mat, frozen_bits, stop_phi):
    """SC 译码至 stop_phi（含判决）。"""
    N = bit_mat.shape[1]
    n = int(math.log2(N))
    loc_row, loc_col = _get_up_loc(bit_mat, n)
    position = [loc_row, loc_col, n, N]

    def decide_left(pos):
        idx = pos[1]
        llr = llr_mat[pos[0] + 1, idx]
        if frozen_bits[idx]:
            return 0
        return 0 if llr >= 0 else 1

    def decide_right(pos):
        idx = pos[1] + 1
        llr = llr_mat[pos[0] + 1, idx]
        if frozen_bits[idx]:
            return 0
        return 0 if llr >= 0 else 1

    def bit_ready(phi):
        v = bit_mat[n, phi]
        return v == 0 or v == 1 or v == 0.0 or v == 1.0

    while not bit_ready(stop_phi):
        span = 2 ** (position[2] - position[0])
        s = position[1]
        up_llr = llr_mat[position[0], s : s + span]
        up_bit = bit_mat[position[0], s : s + span]
        half = span // 2
        left_bit = bit_mat[position[0] + 1, s : s + half]
        right_bit = bit_mat[position[0] + 1, s + half : s + span]

        if _all_filled(up_bit):
            position = _up(position)
            continue
        if _all_filled(right_bit):
            bit_mat[position[0], s : s + span] = _up_bit(
                left_bit.astype(int), right_bit.astype(int)
            )
            continue
        right_llr = llr_mat[position[0] + 1, s + half : s + span]
        left_llr = llr_mat[position[0] + 1, s : s + half]
        if _all_filled(right_llr):
            if position[0] == position[2] - 1:
                bit_mat[position[0] + 1, s + half : s + span] = decide_right(position)
            else:
                position = _rightdown(position)
            continue
        if _all_filled(left_bit):
            llr_mat[position[0] + 1, s + half : s + span] = _right_llr(
                left_bit.astype(int), up_llr
            )
            continue
        if not _all_filled(left_llr):
            llr_mat[position[0] + 1, s : s + half] = _left_llr(up_llr)
            continue
        if position[0] == position[2] - 1:
            bit_mat[position[0] + 1, s : s + half] = decide_left(position)
        else:
            position = _leftdown(position)

    return llr_mat, bit_mat


class SCLDecoder:
    """SCL 译码器（在信息位处分叉）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        rev = bit_reversal_permutation(self.N)
        llr_ch = llr_ch[rev]

        if self.list_size == 1:
            return _sc_decode_core(llr_ch, self.frozen_bits), 0.0

        N = self.N
        n = self.n
        frozen = self.frozen_bits
        info_positions = self.info_indices.tolist()

        llr0 = np.full((n + 1, N), np.nan)
        bit0 = np.full((n + 1, N), np.nan)
        llr0[0] = llr_ch
        paths = [(llr0, bit0, 0.0)]

        prev = -1
        for split_phi in info_positions:
            new_paths = []
            for llr_m, bit_m, pm in paths:
                llr_m, bit_m = llr_m.copy(), bit_m.copy()
                llr_m, bit_m = _sc_step(llr_m, bit_m, frozen, split_phi)
                llr_val = llr_m[n, split_phi]
                if np.isnan(llr_val):
                    u_tmp = _sc_decode_core(llr_m[0], frozen)
                    llr_val = 100.0 if u_tmp[split_phi] == 0 else -100.0

                u_right = int(bit_m[n, split_phi]) if not np.isnan(bit_m[n, split_phi]) else 0
                new_paths.append((llr_m, bit_m, pm + _pm_penalty(llr_val, u_right)))

                bit_wrong = bit_m.copy()
                bit_wrong[n, split_phi] = 1 - u_right
                new_paths.append((llr_m.copy(), bit_wrong, pm + _pm_penalty(llr_val, 1 - u_right)))

            new_paths.sort(key=lambda x: x[2])
            paths = new_paths[: self.list_size]
            prev = split_phi

        for llr_m, bit_m, pm in paths:
            _sc_decode_core(llr_m[0], frozen)

        candidates = []
        for llr_m, bit_m, pm in paths:
            u_hat = bit_m[n].astype(int)
            u_hat[np.isnan(bit_m[n])] = 0
            u_hat = np.nan_to_num(u_hat, nan=0).astype(int)
            full = _sc_decode_core(llr_m[0], frozen)
            for i in range(N):
                v = bit_m[n, i]
                if v == 0 or v == 1:
                    full[i] = int(v)
            candidates.append((full, pm))

        if self.crc_length > 0:
            valid = [
                (u, p)
                for u, p in candidates
                if crc_check(u[self.info_indices], self.crc_length)
            ]
            if valid:
                candidates = valid

        best = min(candidates, key=lambda x: x[1])
        return best[0], best[1]
