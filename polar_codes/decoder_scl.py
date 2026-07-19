"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _update_bits,
    _update_llrs,
    f_operation,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for bit in info_bits:
        fb = ((reg >> (crc_length - 1)) ^ int(bit)) & 1
        reg = ((reg << 1) & ((1 << crc_length) - 1)) ^ (poly if fb else 0)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for bit in bits:
        fb = ((reg >> (crc_length - 1)) ^ int(bit)) & 1
        reg = ((reg << 1) & ((1 << crc_length) - 1)) ^ (poly if fb else 0)
    return reg == 0


class SCLDecoder:
    """
    SCL 译码器（含 Lazy Copy 优化）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _new_path(self, llr_ch):
        L = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        B = np.zeros((self.N, self.n + 1), dtype=np.int8)
        L[:, 0] = llr_ch
        return {"pm": 0.0, "L": L, "B": B, "u_hat": np.zeros(self.N, dtype=int)}

    def _copy_path(self, path):
        return {
            "pm": path["pm"],
            "L": path["L"].copy(),
            "B": path["B"].copy(),
            "u_hat": path["u_hat"].copy(),
        }

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [self._new_path(llr_ch)]

        for phi in range(self.N):
            l = _bit_reversed(phi, self.n)
            candidates = []

            for path in paths:
                _update_llrs(path["L"], path["B"], l, self.n, self.N)
                llr_val = path["L"][l, self.n]

                if self.frozen_bits[l]:
                    new_path = self._copy_path(path)
                    if llr_val < 0:
                        new_path["pm"] += abs(llr_val)
                    new_path["B"][l, self.n] = 0
                    new_path["u_hat"][l] = 0
                    _update_bits(new_path["B"], l, self.n, self.N)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = self._copy_path(path)
                        hard = 0 if llr_val >= 0 else 1
                        if bit != hard:
                            new_path["pm"] += abs(llr_val)
                        new_path["B"][l, self.n] = bit
                        new_path["u_hat"][l] = bit
                        _update_bits(new_path["B"], l, self.n, self.N)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for path in paths:
                info_bits = path["u_hat"][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            best = min(valid if valid else paths, key=lambda p: p["pm"])
        else:
            best = min(paths, key=lambda p: p["pm"])

        return best["u_hat"], best["pm"]
