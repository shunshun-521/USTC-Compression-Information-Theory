"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math
import copy

from decoder_sc import _SCDEngine, _bit_reversed, _as_frozen_mask, channel_llr_to_decoder


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0xE0  # bit-reversed CRC-8 (0x07)
    if crc_length == 16:
        return 0xA001  # bit-reversed CRC-16 (0x8005)
    raise ValueError("crc_length must be 8 or 16")


def _crc_bits(data_bits, crc_length, poly):
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in data_bits:
        fb = ((reg >> (crc_length - 1)) & 1) ^ int(bit)
        reg = (reg << 1) & mask
        if fb:
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int64)
    poly = _crc_poly(crc_length)
    padded = list(info_bits) + [0] * crc_length
    reg = _crc_bits(padded, crc_length, poly)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int64
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int64)
    poly = _crc_poly(crc_length)
    return _crc_bits(bits, crc_length, poly) == 0


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = _as_frozen_mask(frozen_bits, N)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = None if info_indices is None else np.asarray(info_indices, dtype=int)

    def decode(self, llr_ch):
        llr_ch = channel_llr_to_decoder(llr_ch)
        n = self.n

        paths = [{"engine": _SCDEngine(self.N, self.frozen_bits), "pm": 0.0, "u": np.zeros(self.N, int)}]
        paths[0]["engine"].L[:, 0] = llr_ch

        for i in range(self.N):
            l = _bit_reversed(i, n)
            new_paths = []
            for path in paths:
                eng = path["engine"]
                eng.update_llrs(l)
                llr = eng.L[l, n]
                if np.isnan(llr):
                    llr = 0.0

                def extend(bit, pm_base, llr_val):
                    child = {
                        "engine": copy.deepcopy(eng),
                        "pm": pm_base,
                        "u": path["u"].copy(),
                    }
                    child["engine"].B[l, n] = bit
                    child["u"][l] = bit
                    child["engine"].update_bits(l)
                    return child

                if l in self.frozen_set:
                    penalty = 0.0 if llr >= 0 else abs(llr)
                    new_paths.append(extend(0, path["pm"] + penalty, llr))
                else:
                    exp_bit = 0 if llr >= 0 else 1
                    for bit in (0, 1):
                        penalty = 0.0 if bit == exp_bit else abs(llr)
                        new_paths.append(extend(bit, path["pm"] + penalty, llr))

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                if self.info_indices is None:
                    payload = p["u"][: len(p["u"])]
                else:
                    payload = p["u"][self.info_indices]
                if crc_check(payload, self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p["pm"])
        return best["u"], best["pm"]
