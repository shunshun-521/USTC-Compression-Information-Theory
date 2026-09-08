"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _update_bits,
    _update_llrs,
    sc_decode,
)
from encoder import bit_reversal_permutation


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    """Reflected CRC remainder (poly 0x07 for CRC-8, 0x8005 for CRC-16)."""
    crc = 0
    mask = (1 << crc_length) - 1
    for bit in bits:
        crc ^= int(bit)
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ poly
            else:
                crc >>= 1
    return crc & mask


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 是否通过 CRC"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class SCLDecoder:
    """SCL 译码器（路径复制实现）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.br = bit_reversal_permutation(N)

    def _pm_update(self, pm, llr, bit):
        hard = 0 if llr >= 0 else 1
        if bit != hard:
            pm += abs(llr)
        return pm

    def decode(self, llr_ch):
        """主译码函数"""
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        N = self.N
        n = self.n

        paths = [
            {
                "L": np.zeros((N, n + 1), dtype=np.float64),
                "B": np.zeros((N, n + 1), dtype=np.int32),
                "pm": 0.0,
                "u_hat": np.zeros(N, dtype=np.int32),
            }
        ]
        paths[0]["L"][:, 0] = llr_ch.astype(np.float64)

        for phi in range(N):
            l = self.br[phi]
            new_paths = []

            for path in paths:
                _update_llrs(path["L"], path["B"], l, n)
                llr_bit = path["L"][l, n]

                if self.frozen_bits[phi]:
                    new_path = {
                        "L": path["L"].copy(),
                        "B": path["B"].copy(),
                        "pm": self._pm_update(path["pm"], llr_bit, 0),
                        "u_hat": path["u_hat"].copy(),
                    }
                    new_path["B"][l, n] = 0
                    new_path["u_hat"][phi] = 0
                    _update_bits(new_path["B"], l, n, N)
                    new_paths.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = {
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                            "pm": self._pm_update(path["pm"], llr_bit, bit),
                            "u_hat": path["u_hat"].copy(),
                        }
                        new_path["B"][l, n] = bit
                        new_path["u_hat"][phi] = bit
                        _update_bits(new_path["B"], l, n, N)
                        new_paths.append(new_path)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for path in paths:
                info_bits = path["u_hat"][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            if valid:
                paths = valid

        best = paths[0]
        return best["u_hat"].copy(), best["pm"]
