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
    _lower_llr,
    _update_bits,
    _update_llrs,
    _upper_llr,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_poly(crc_length):
    if crc_length == 8:
        return CRC8_POLY
    if crc_length == 16:
        return CRC16_POLY
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_poly(crc_length)
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 LLR/比特数组）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length

    def _path_penalty(self, llr, u_bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr)

    def _advance_path(self, L, B, l, u_bit):
        if l in self.frozen_set:
            u_bit = 0
        B[l, self.n] = u_bit
        _update_bits(B, l, self.n, self.N)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        L0 = np.full((N, n + 1), np.nan, dtype=np.float64)
        B0 = np.full((N, n + 1), np.nan)
        L0[:, 0] = llr_ch
        paths = [{"pm": 0.0, "L": L0, "B": B0, "u_hat": np.zeros(N, dtype=int)}]

        for i in range(N):
            l = _bit_reversed(i, n)
            candidates = []
            for path in paths:
                L = path["L"]
                B = path["B"]
                _update_llrs(L, B, l, n)
                llr = L[l, n]

                if l in self.frozen_set:
                    pm = path["pm"] + self._path_penalty(llr, 0)
                    new_path = {
                        "pm": pm,
                        "L": copy.deepcopy(L),
                        "B": copy.deepcopy(B),
                        "u_hat": path["u_hat"].copy(),
                    }
                    new_path["u_hat"][l] = 0
                    self._advance_path(new_path["L"], new_path["B"], l, 0)
                    candidates.append(new_path)
                else:
                    for u_bit in (0, 1):
                        pm = path["pm"] + self._path_penalty(llr, u_bit)
                        new_path = {
                            "pm": pm,
                            "L": copy.deepcopy(L),
                            "B": copy.deepcopy(B),
                            "u_hat": path["u_hat"].copy(),
                        }
                        new_path["u_hat"][l] = u_bit
                        self._advance_path(new_path["L"], new_path["B"], l, u_bit)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            crc_paths = [p for p in paths if crc_check(p["u_hat"], self.crc_length)]
            pool = crc_paths if crc_paths else paths
        else:
            pool = paths

        best = min(pool, key=lambda p: p["pm"])
        return best["u_hat"].copy(), best["pm"]
