"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _pm_penalty,
    _update_bits,
    _update_llrs,
)
from encoder import bit_reversed_index


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8 if crc_length == 8 else 1):
            if crc_length == 8:
                if reg & 0x80:
                    reg = ((reg << 1) ^ (poly << (crc_length - 8))) & 0xFF
                else:
                    reg = (reg << 1) & 0xFF
            else:
                if reg & (1 << (crc_length - 1)):
                    reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
                else:
                    reg = (reg << 1) & ((1 << crc_length) - 1)
    if crc_length == 8:
        crc_bits = np.array([(reg >> i) & 1 for i in range(7, -1, -1)], dtype=int)
    else:
        crc_bits = np.array([(reg >> i) & 1 for i in range(15, -1, -1)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 的 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    return np.array_equal(bits, crc_encode(bits[:-crc_length], crc_length))


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径共享 L/B 数组，分裂时按需复制）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.decode_order = [bit_reversed_index(i, self.n) for i in range(N)]

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        frozen_set = set(np.where(self.frozen_bits)[0])

        paths = [{
            "pm": 0.0,
            "L": np.full((self.N, self.n + 1), np.nan, dtype=np.float64),
            "B": np.full((self.N, self.n + 1), np.nan),
            "u_hat": np.zeros(self.N, dtype=np.int32),
        }]
        paths[0]["L"][:, 0] = llr_ch

        for l in self.decode_order:
            candidates = []
            for path in paths:
                _update_llrs(path["L"], path["B"], l, self.n)
                cur_llr = path["L"][l, self.n]

                if l in frozen_set:
                    new_path = self._clone_path(path)
                    new_path["pm"] += _pm_penalty(cur_llr, 0)
                    new_path["B"][l, self.n] = 0
                    new_path["u_hat"][l] = 0
                    _update_bits(new_path["B"], l, self.n)
                    candidates.append(new_path)
                else:
                    for u in (0, 1):
                        new_path = self._clone_path(path)
                        new_path["pm"] += _pm_penalty(cur_llr, u)
                        new_path["B"][l, self.n] = u
                        new_path["u_hat"][l] = u
                        _update_bits(new_path["B"], l, self.n)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        best = paths[0]
        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p["u_hat"][self.info_indices], self.crc_length)]
            if valid:
                best = min(valid, key=lambda p: p["pm"])

        return best["u_hat"].copy(), best["pm"]

    @staticmethod
    def _clone_path(path):
        return {
            "pm": path["pm"],
            "L": path["L"].copy(),
            "B": path["B"].copy(),
            "u_hat": path["u_hat"].copy(),
        }
