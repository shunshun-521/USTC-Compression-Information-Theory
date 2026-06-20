"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import sc_decode, sc_stepping_decode


CRC_POLYS = {8: 0x07, 16: 0x8005}


def _crc_process(bits, crc_length):
    poly = CRC_POLYS[crc_length]
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    return np.concatenate([info_bits, _crc_process(info_bits, crc_length)])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    return np.array_equal(bits[-crc_length:], _crc_process(bits[:-crc_length], crc_length))


def _path_metric_update(llrs, bits):
    pm = 0.0
    for llr, bit in zip(llrs, bits):
        hard = 0 if llr >= 0 else 1
        if bit != hard:
            pm += abs(llr)
    return pm


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = (
            np.asarray(info_indices, dtype=int)
            if info_indices is not None
            else np.where(~self.frozen_bits)[0]
        )

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        llr0 = np.full((n + 1, N), np.nan, dtype=np.float64)
        bit0 = np.full((n + 1, N), np.nan, dtype=np.float64)
        llr0[0, :] = llr_ch

        paths = [(llr0, bit0, 0.0)]
        prev_pos = -1

        for split_pos in range(N):
            new_paths = []
            for llr_m, bit_m, pm in paths:
                llr_m, bit_m = sc_stepping_decode(llr_m, bit_m, self.frozen_bits, split_pos)

                if self.frozen_bits[split_pos]:
                    seg_llr = llr_m[n, prev_pos + 1 : split_pos + 1]
                    seg_bit = bit_m[n, prev_pos + 1 : split_pos + 1].astype(int)
                    new_pm = pm + _path_metric_update(seg_llr, seg_bit)
                    new_paths.append((llr_m, bit_m, new_pm))
                else:
                    base_bit = int(bit_m[n, split_pos])
                    for bit_val in (base_bit, 1 - base_bit):
                        llr_c = llr_m.copy()
                        bit_c = bit_m.copy()
                        bit_c[n, split_pos] = bit_val
                        seg_llr = llr_c[n, prev_pos + 1 : split_pos + 1]
                        seg_bit = bit_c[n, prev_pos + 1 : split_pos + 1].astype(int)
                        seg_bit[-1] = bit_val
                        new_pm = pm + _path_metric_update(seg_llr, seg_bit)
                        new_paths.append((llr_c, bit_c, new_pm))

            new_paths.sort(key=lambda item: item[2])
            paths = new_paths[: self.list_size]
            prev_pos = split_pos

        if self.crc_length > 0:
            valid = []
            for llr_m, bit_m, pm in paths:
                info_bits = bit_m[n, self.info_indices].astype(int)
                if crc_check(info_bits, self.crc_length):
                    valid.append((bit_m, pm))
            if valid:
                best_bits, best_pm = min(valid, key=lambda x: x[1])
            else:
                best_bits, _, best_pm = paths[0]
            return best_bits[n, :].astype(int), float(best_pm)

        _, best_bits, best_pm = min(paths, key=lambda item: item[2])
        return best_bits[n, :].astype(int), float(best_pm)
