"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from sc_stepping import _sc_step_once, sc_decode_stepping


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in info_bits:
        feedback = ((reg >> (crc_length - 1)) ^ int(bit)) & 1
        reg = (reg << 1) & mask
        if feedback:
            reg ^= poly
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


def _pm_update(llr_leaf, bit_val):
    hard = 0 if llr_leaf >= 0 else 1
    if bit_val == hard:
        return 0.0
    return abs(llr_leaf)


def _init_matrices(y_llr):
    N = y_llr.size
    n = int(np.log2(N))
    llr_matrix = np.ones((n + 1, N), dtype=np.float64)
    llr_matrix[llr_matrix == 1] = np.nan
    bit_matrix = llr_matrix.copy()
    llr_matrix[0] = y_llr
    return llr_matrix, bit_matrix


class SCLDecoder:
    """SCL 译码器（基于因子图步进 SC）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def decode(self, llr_ch):
        y_llr = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1:
            u_hat = sc_decode_stepping(y_llr, self.frozen_set)
            return u_hat, 0.0

        llr0, bit0 = _init_matrices(y_llr)
        paths = [(llr0.copy(), bit0.copy(), 0.0)]
        split_positions = [i for i in self.info_indices]

        for split_pos in split_positions:
            new_paths = []
            for llr_m, bit_m, pm in paths:
                llr_m, bit_m = _sc_step_once(llr_m.copy(), bit_m.copy(), self.frozen_set, split_pos)
                llr_leaf = llr_m[self.n, split_pos]
                u_bit = int(bit_m[self.n, split_pos])
                pm_new = pm + _pm_update(llr_leaf, u_bit)
                new_paths.append((llr_m, bit_m, pm_new))
                if split_pos not in self.frozen_set:
                    bit_wrong = bit_m.copy()
                    bit_wrong[self.n, split_pos] = 1 - u_bit
                    pm_wrong = pm + _pm_update(llr_leaf, 1 - u_bit)
                    new_paths.append((llr_m.copy(), bit_wrong, pm_wrong))

            new_paths.sort(key=lambda x: x[2])
            paths = new_paths[: self.list_size]

        llr_m, bit_m, pm = paths[0]
        llr_m, bit_m = _sc_step_once(llr_m, bit_m, self.frozen_set, self.N - 1)
        candidates = [(bit_m[self.n].astype(int), pm)] + [
            (p[1][self.n].astype(int), p[2]) for p in paths[1:]
        ]

        if self.crc_length > 0:
            valid = []
            for u_hat, path_pm in candidates:
                payload = u_hat[self.info_indices]
                if crc_check(payload, self.crc_length):
                    valid.append((u_hat, path_pm))
            if valid:
                best = min(valid, key=lambda x: x[1])[0]
                return best, valid[0][1]

        best = min(candidates, key=lambda x: x[1])[0]
        return best, min(candidates, key=lambda x: x[1])[1]
