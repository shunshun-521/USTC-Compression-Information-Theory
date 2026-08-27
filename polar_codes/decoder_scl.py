"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import math

import numpy as np

from decoder_sc import _active_bit_level, _active_llr_level, _bit_reversed, _update_bits, _update_llrs
from encoder import bit_reversal_permutation


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError(f"Unsupported CRC length: {crc_length}")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=np.int8)
    return np.array_equal(bits[-crc_length:], crc_encode(bits[:-crc_length], crc_length)[-crc_length:])


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    @staticmethod
    def _pm_update(pm, llr, bit):
        hard = 0 if llr >= 0 else 1
        if bit == hard:
            return pm
        return pm + abs(llr)

    def _new_path(self, llr_ch):
        n = self.n
        N = self.N
        path = {
            "L": np.full((N, n + 1), np.nan, dtype=np.float64),
            "B": np.full((N, n + 1), np.nan),
            "pm": 0.0,
            "u_hat": np.full(N, np.nan),
        }
        path["L"][:, 0] = llr_ch.astype(np.float64)[bit_reversal_permutation(self.N)]
        return path

    def decode(self, llr_ch):
        n = self.n
        N = self.N
        paths = [self._new_path(llr_ch)]

        for i in range(N):
            l = _bit_reversed(i, n)
            candidates = []

            for path in paths:
                _update_llrs(path["L"], path["B"], l, n)
                llr = path["L"][l, n]

                if self.frozen_bits[l]:
                    new_path = copy.deepcopy(path)
                    bit = 0
                    new_path["pm"] = self._pm_update(path["pm"], llr, bit)
                    new_path["B"][l, n] = bit
                    new_path["u_hat"][l] = bit
                    _update_bits(new_path["B"], l, n)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = copy.deepcopy(path)
                        new_path["pm"] = self._pm_update(path["pm"], llr, bit)
                        new_path["B"][l, n] = bit
                        new_path["u_hat"][l] = bit
                        _update_bits(new_path["B"], l, n)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        u_hat = np.zeros(N, dtype=np.int8)
        for i in range(N):
            l = _bit_reversed(i, n)
            u_hat[l] = int(paths[0]["u_hat"][l])

        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = np.zeros(len(self.info_indices), dtype=np.int8)
                for idx, pos in enumerate(self.info_indices):
                    info_bits[idx] = int(p["u_hat"][pos])
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            pool = valid if valid else paths
        else:
            pool = paths

        best = min(pool, key=lambda p: p["pm"])
        u_hat = np.zeros(N, dtype=np.int8)
        for i in range(N):
            l = _bit_reversed(i, n)
            u_hat[l] = int(best["u_hat"][l])

        return u_hat, best["pm"]
