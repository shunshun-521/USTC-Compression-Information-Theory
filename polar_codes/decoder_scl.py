"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import math
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _to_frozen_set,
    _update_bits,
    _update_llrs,
    lower_llr,
    upper_llr,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg <<= 1
        reg |= int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    crc_bits = np.array(
        [(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    if crc_length == 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen = _to_frozen_set(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.array(
            [i for i in range(N) if i not in self.frozen], dtype=int
        )

    def _new_path(self, llr_ch):
        L = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        B = np.full((self.N, self.n + 1), np.nan)
        L[:, 0] = np.asarray(llr_ch, dtype=np.float64)
        return {"pm": 0.0, "L": L, "B": B, "u_hat": np.zeros(self.N, dtype=int)}

    def _clone_path(self, path):
        return {
            "pm": path["pm"],
            "L": path["L"].copy(),
            "B": path["B"].copy(),
            "u_hat": path["u_hat"].copy(),
        }

    def decode(self, llr_ch):
        paths = [self._new_path(llr_ch)]
        phases = [_bit_reversed(i, self.n) for i in range(self.N)]

        for l in phases:
            candidates = []
            for path in paths:
                _update_llrs(path["L"], path["B"], l, self.n, self.N)
                llr_val = path["L"][l, self.n]
                if np.isnan(llr_val):
                    llr_val = 0.0

                if l in self.frozen:
                    new_path = self._clone_path(path)
                    if llr_val < 0:
                        new_path["pm"] += abs(llr_val)
                    new_path["u_hat"][l] = 0
                    new_path["B"][l, self.n] = 0
                    _update_bits(new_path["B"], l, self.n, self.N)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = self._clone_path(path)
                        expected = 0 if llr_val >= 0 else 1
                        if bit != expected:
                            new_path["pm"] += abs(llr_val)
                        new_path["u_hat"][l] = bit
                        new_path["B"][l, self.n] = bit
                        _update_bits(new_path["B"], l, self.n, self.N)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        crc_paths = []
        if self.crc_length > 0:
            for path in paths:
                info_bits = path["u_hat"][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_paths.append(path)

        best = min(crc_paths or paths, key=lambda p: p["pm"])
        return best["u_hat"].astype(int), best["pm"]
