"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math
import copy

from encoder import bit_reversal_permutation
from decoder_sc import (
    f_operation,
    g_operation,
    _active_llr_level,
    _active_bit_level,
    _update_llrs,
    _update_bits,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY

    if crc_length == 8:
        reg = 0
        for bit in info_bits:
            reg ^= int(bit) << 7
            for _ in range(8):
                if reg & 0x80:
                    reg = ((reg << 1) ^ poly) & 0xFF
                else:
                    reg = (reg << 1) & 0xFF
        crc_bits = np.array([(reg >> (7 - i)) & 1 for i in range(8)], dtype=int)
    else:
        reg = 0
        for bit in info_bits:
            reg ^= int(bit) << 15
            for _ in range(16):
                if reg & 0x8000:
                    reg = ((reg << 1) ^ poly) & 0xFFFF
                else:
                    reg = (reg << 1) & 0xFFFF
        crc_bits = np.array([(reg >> (15 - i)) & 1 for i in range(16)], dtype=int)

    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    encoded = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(encoded, bits)


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.asarray(info_indices, dtype=int) if info_indices is not None else None
        self.br = bit_reversal_permutation(N)
        self.br_order = [int(self.br[i]) for i in range(N)]

    def _path_penalty(self, llr, u):
        preferred = 0 if llr >= 0 else 1
        return 0.0 if u == preferred else abs(llr)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, pm
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)[self.br]
        n, N = self.n, self.N

        paths = [{
            "pm": 0.0,
            "L": np.zeros((N, n + 1), dtype=np.float64),
            "B": np.zeros((N, n + 1), dtype=int),
            "u_hat": np.zeros(N, dtype=int),
        }]
        paths[0]["L"][:, 0] = llr_ch

        for l in self.br_order:
            candidates = []
            for path in paths:
                _update_llrs(path["L"], path["B"], l, n)
                llr = path["L"][l, n]

                if self.frozen_bits[l]:
                    pm = path["pm"] + self._path_penalty(llr, 0)
                    new_path = {
                        "pm": pm,
                        "L": path["L"].copy(),
                        "B": path["B"].copy(),
                        "u_hat": path["u_hat"].copy(),
                    }
                    new_path["B"][l, n] = 0
                    _update_bits(new_path["B"], l, n)
                    new_path["u_hat"][l] = 0
                    candidates.append(new_path)
                else:
                    for u in (0, 1):
                        pm = path["pm"] + self._path_penalty(llr, u)
                        new_path = {
                            "pm": pm,
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                            "u_hat": path["u_hat"].copy(),
                        }
                        new_path["B"][l, n] = u
                        _update_bits(new_path["B"], l, n)
                        new_path["u_hat"][l] = u
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                bits = p["u_hat"][self.info_indices] if self.info_indices is not None else p["u_hat"]
                if crc_check(bits, self.crc_length):
                    valid.append(p)
            best = min(valid, key=lambda p: p["pm"]) if valid else min(paths, key=lambda p: p["pm"])
        else:
            best = min(paths, key=lambda p: p["pm"])

        return best["u_hat"], best["pm"]
