"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation
from decoder_sc import (
    bit_reversed_index,
    path_metric_update,
    _update_llrs,
    _update_bits,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_encode_bits(info_bits, poly, crc_length):
    reg = 0
    top = 1 << (crc_length - 1)
    mask = (1 << crc_length) - 1
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & top:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    crc_bits = _crc_encode_bits(info_bits, poly, crc_length)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length=crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


class SCLDecoder:
    """SCL 译码器（Permuted SC + 路径复制）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int).astype(bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.decode_order = [bit_reversed_index(i, self.n) for i in range(N)]
        self.perm = bit_reversal_permutation(N)

    def _init_path(self, llr_ch):
        path = {
            "pm": 0.0,
            "L": np.zeros((self.N, self.n + 1), dtype=np.float64),
            "B": np.zeros((self.N, self.n + 1), dtype=np.int8),
            "u_hat": np.zeros(self.N, dtype=int),
        }
        path["L"][:, 0] = llr_ch[self.perm]
        return path

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [self._init_path(llr_ch)]

        for l in self.decode_order:
            candidates = []
            for path in paths:
                _update_llrs(path["L"], path["B"], l, self.n)
                llr0 = path["L"][l, self.n]

                branches = [0] if l in self.frozen_set else [0, 1]
                for u in branches:
                    child = {
                        "pm": path_metric_update(path["pm"], llr0, u),
                        "L": path["L"].copy(),
                        "B": path["B"].copy(),
                        "u_hat": path["u_hat"].copy(),
                    }
                    child["B"][l, self.n] = u
                    child["u_hat"][l] = u
                    _update_bits(child["B"], l, self.n)
                    candidates.append(child)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        best = paths[0]
        if self.crc_length > 0:
            info_positions = np.where(~self.frozen_bits)[0]
            valid = [p for p in paths if crc_check(p["u_hat"][info_positions], self.crc_length)]
            if valid:
                best = min(valid, key=lambda p: p["pm"])

        return best["u_hat"].copy(), best["pm"]
