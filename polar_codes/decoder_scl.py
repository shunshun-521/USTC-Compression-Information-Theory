"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import _sc_decode_core, sc_decode
from encoder import bit_reversal_permutation

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([
        (remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)
    ], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class SCLDecoder:
    """SCL 译码器（路径列表 + 路径度量）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.info_indices = np.where(self.frozen_bits == 0)[0]
        self.rev = bit_reversal_permutation(N)
        self.info_br = self.rev[self.info_indices]
        self.list_size = list_size
        self.crc_length = crc_length

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        from decoder_sc import (
            _all_filled, _get_left_bit, _get_left_llr, _get_right_bit,
            _get_right_llr, _get_up_bit, _leftdown, _rightdown, _up,
        )

        info_set = set(int(i) for i in self.info_br)

        def init_matrices():
            llr_m = np.full((n + 1, N), np.nan)
            bit_m = np.full((n + 1, N), np.nan)
            llr_m[0] = llr_ch
            return llr_m, bit_m

        def step_to(split_pos, llr_m, bit_m):
            loc = [0, 0]
            for i in range(n + 1):
                for j in range(N):
                    if np.isnan(bit_m[i, j]):
                        loc = [i, j]
                        break
                else:
                    continue
                break
            position = [loc[0], loc[1], n, N]
            while np.isnan(bit_m[n, split_pos]):
                span = 2 ** (position[2] - position[0])
                up_llr = llr_m[position[0]][position[1]:position[1] + span]
                up_bit = bit_m[position[0]][position[1]:position[1] + span]
                left_llr = llr_m[position[0] + 1][position[1]:position[1] + span // 2]
                left_bit = bit_m[position[0] + 1][position[1]:position[1] + span // 2]
                right_llr = llr_m[position[0] + 1][position[1] + span // 2:position[1] + span]
                right_bit = bit_m[position[0] + 1][position[1] + span // 2:position[1] + span]

                if _all_filled(up_bit):
                    position = _up(position)
                elif _all_filled(right_bit):
                    val = _get_up_bit(left_bit, right_bit)
                    bit_m[position[0]][position[1]:position[1] + span] = val.copy()
                elif _all_filled(right_llr):
                    if position[0] == position[2] - 1:
                        bit_m[position[0] + 1][position[1] + span // 2:position[1] + span] = (
                            _get_right_bit(right_llr, info_set, position[1] + 1)
                        )
                    else:
                        position = _rightdown(position)
                elif _all_filled(left_bit):
                    llr_m[position[0] + 1][position[1] + span // 2:position[1] + span] = (
                        _get_right_llr(left_bit, up_llr)
                    )
                elif not _all_filled(left_llr):
                    llr_m[position[0] + 1][position[1]:position[1] + span // 2] = _get_left_llr(up_llr)
                else:
                    if position[0] == position[2] - 1:
                        bit_m[position[0] + 1][position[1]:position[1] + span // 2] = (
                            _get_left_bit(left_llr, info_set, position[1])
                        )
                    else:
                        position = _leftdown(position)
            return llr_m, bit_m

        llr_list, bit_list, pm_list = [], [], []
        lm, bm = init_matrices()
        llr_list.append(lm)
        bit_list.append(bm)
        pm_list.append(0.0)

        prev = -1
        split_positions = list(self.info_br)
        for split_pos in split_positions:
            new_llr, new_bit, new_pm = [], [], []
            for llr_m, bit_m, pm in zip(llr_list, bit_list, pm_list):
                llr_m, bit_m = step_to(split_pos, llr_m.copy(), bit_m.copy())
                llr_val = llr_m[n, split_pos]
                for bit in (0, 1):
                    bm2 = bit_m.copy()
                    bm2[n, split_pos] = bit
                    new_llr.append(llr_m.copy())
                    new_bit.append(bm2)
                    new_pm.append(pm + self._pm_penalty(llr_val, bit))
            order = np.argsort(new_pm)[:self.list_size]
            llr_list = [new_llr[i] for i in order]
            bit_list = [new_bit[i] for i in order]
            pm_list = [new_pm[i] for i in order]
            prev = split_pos

        if split_positions and split_positions[-1] != N - 1:
            for i in range(len(llr_list)):
                llr_list[i], bit_list[i] = step_to(N - 1, llr_list[i], bit_list[i])

        candidates = []
        for bit_m, pm in zip(bit_list, pm_list):
            u_br = bit_m[n].astype(int)
            u = self.rev[u_br]
            if self.crc_length > 0:
                if crc_check(u[self.info_indices], self.crc_length):
                    candidates.append((pm, u))
            else:
                candidates.append((pm, u))

        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1].copy(), candidates[0][0]

        best = int(np.argmin(pm_list))
        u_br = bit_list[best][n].astype(int)
        return self.rev[u_br].copy(), pm_list[best]


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(1)
    mismatches = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.01)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    print(f"SCL L=1 vs SC mismatches: {mismatches}/50")
