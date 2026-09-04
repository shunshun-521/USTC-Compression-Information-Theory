"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    sc_decode,
    _all_decided,
    _leftdown,
    _rightdown,
    _up,
    _get_up_bit,
    _get_left_llr,
    _get_right_llr,
    _decide_bit,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, width, poly):
    reg = 0
    full_poly = (1 << width) | poly
    for bit in bits:
        reg = ((reg << 1) | int(bit)) & ((1 << (width + 1)) - 1)
        if reg & (1 << width):
            reg ^= full_poly
    return reg & ((1 << width) - 1)


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    padded = np.concatenate([info_bits, np.zeros(crc_length, dtype=int)])
    remainder = _crc_remainder(padded, crc_length, poly)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, crc_length, poly) == 0


def _pm_penalty(llr, bit):
    hard = 0 if llr >= 0 else 1
    return 0.0 if bit == hard else abs(llr)


def _sc_step_to_bit(llr_matrix, bit_matrix, info_positions, frozen_value, target_bit):
    """将 SC 译码推进到 target_bit 位置并完成该位判决。"""
    N = llr_matrix.shape[1]
    n = int(np.log2(N))
    position = [0, 0, n, N]

    while not _all_decided(bit_matrix[n]):
        span = 2 ** (position[2] - position[0])
        up_llr = llr_matrix[position[0]][position[1]: position[1] + span]
        up_bit = bit_matrix[position[0]][position[1]: position[1] + span]
        left_llr = llr_matrix[position[0] + 1][position[1]: position[1] + span // 2]
        left_bit = bit_matrix[position[0] + 1][position[1]: position[1] + span // 2]
        right_llr = llr_matrix[position[0] + 1][position[1] + span // 2: position[1] + span]
        right_bit = bit_matrix[position[0] + 1][position[1] + span // 2: position[1] + span]

        if _all_decided(up_bit):
            position = _up(position)
        elif _all_decided(right_bit):
            bit_matrix[position[0]][position[1]: position[1] + span] = _get_up_bit(left_bit, right_bit)
        elif _all_decided(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                is_info = right_bit_pos in info_positions
                val = _decide_bit(right_llr[0], not is_info, frozen_value)
                bit_matrix[position[0] + 1][position[1] + span // 2: position[1] + span] = val
                if right_bit_pos == target_bit:
                    return llr_matrix, bit_matrix, right_llr[0]
            else:
                position = _rightdown(position)
        elif _all_decided(left_bit):
            llr_matrix[position[0] + 1][position[1] + span // 2: position[1] + span] = _get_right_llr(
                left_bit, up_llr
            )
        elif not _all_decided(left_llr):
            llr_matrix[position[0] + 1][position[1]: position[1] + span // 2] = _get_left_llr(up_llr)
        else:
            if position[0] == position[2] - 1:
                left_bit_pos = position[1]
                is_info = left_bit_pos in info_positions
                val = _decide_bit(left_llr[0], not is_info, frozen_value)
                bit_matrix[position[0] + 1][position[1]: position[1] + span // 2] = val
                if left_bit_pos == target_bit:
                    return llr_matrix, bit_matrix, left_llr[0]
            else:
                position = _leftdown(position)

    return llr_matrix, bit_matrix, 0.0


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(self.frozen_bits == 0)[0]
        self.frozen_value = 0

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        if self.list_size == 1:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr_matrix = np.full((n + 1, N), np.nan)
        bit_matrix = np.full((n + 1, N), np.nan)
        llr_matrix[0] = llr_ch

        paths = [(llr_matrix.copy(), bit_matrix.copy(), 0.0)]

        for bit_pos in range(N):
            new_paths = []
            is_frozen = self.frozen_bits[bit_pos] == 1

            for llr_m, bit_m, pm in paths:
                if _all_decided(bit_m[n]):
                    new_paths.append((llr_m, bit_m, pm))
                    continue

                llr_m2, bit_m2, cur_llr = _sc_step_to_bit(
                    llr_m.copy(), bit_m.copy(), self.info_positions, self.frozen_value, bit_pos
                )

                if is_frozen:
                    new_paths.append((llr_m2, bit_m2, pm + _pm_penalty(cur_llr, 0)))
                else:
                    for b in (0, 1):
                        lm = llr_m2.copy()
                        bm = bit_m2.copy()
                        bm[n][bit_pos] = b
                        new_paths.append((lm, bm, pm + _pm_penalty(cur_llr, b)))

            new_paths.sort(key=lambda x: x[2])
            paths = new_paths[: self.list_size]

        best_crc = None
        best_crc_pm = float("inf")
        best = None
        best_pm = float("inf")

        for llr_m, bit_m, pm in paths:
            u_hat = bit_m[n].astype(int)
            if pm < best_pm:
                best_pm = pm
                best = u_hat
            if self.crc_length > 0:
                payload = u_hat[self.info_positions]
                if crc_check(payload, self.crc_length) and pm < best_crc_pm:
                    best_crc_pm = pm
                    best_crc = u_hat

        chosen = best_crc if best_crc is not None else best
        return chosen.copy(), (best_crc_pm if best_crc is not None else best_pm)
