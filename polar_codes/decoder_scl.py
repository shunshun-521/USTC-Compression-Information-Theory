"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    sc_decode,
    upper_llr,
    lower_llr,
    active_llr_level,
    active_bit_level,
    bit_reversed,
    _update_bits,
)


def _crc8_remainder(bits):
    reg = 0
    for bit in bits:
        reg ^= int(bit) << 7
        for _ in range(8):
            if reg & 0x80:
                reg = ((reg << 1) ^ 0x07) & 0xFF
            else:
                reg = (reg << 1) & 0xFF
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        reg = 0
        for bit in info_bits:
            reg ^= int(bit) << 7
            for _ in range(8):
                if reg & 0x80:
                    reg = ((reg << 1) ^ 0x07) & 0xFF
                else:
                    reg = (reg << 1) & 0xFF
        crc_bits = np.array([(reg >> i) & 1 for i in range(7, -1, -1)], dtype=np.int8)
        return np.concatenate([info_bits, crc_bits])
    if crc_length == 16:
        poly = 0x8005
        reg = 0
        for bit in info_bits:
            reg ^= int(bit) << 15
            for _ in range(1):
                if reg & 0x8000:
                    reg = ((reg << 1) ^ poly) & 0xFFFF
                else:
                    reg = (reg << 1) & 0xFFFF
        crc_bits = np.array([(reg >> i) & 1 for i in range(15, -1, -1)], dtype=np.int8)
        return np.concatenate([info_bits, crc_bits])
    raise ValueError("crc_length must be 8 or 16")


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    if crc_length == 8:
        return _crc8_remainder(bits) == 0
    if crc_length == 16:
        poly = 0x8005
        reg = 0
        for bit in bits:
            reg ^= int(bit) << 15
            for _ in range(1):
                if reg & 0x8000:
                    reg = ((reg << 1) ^ poly) & 0xFFFF
                else:
                    reg = (reg << 1) & 0xFFFF
        return reg == 0
    return False


def _pm_update(pm, llr, u):
    hard = 0 if llr >= 0 else 1
    return pm if u == hard else pm + abs(llr)


def _update_llrs_path(L, B, l, n):
    N = L.shape[0]
    for s in range(n - active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = lower_llr(L[j, s], L[j - branch_size, s], int(B[j - branch_size, s + 1]))


class SCLDecoder:
    """SCL 译码器（路径分裂时复制 L/B 数组）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        assert 2 ** self.n == N
        fb = np.asarray(frozen_bits)
        self.frozen = set(np.where(fb.astype(bool) if fb.dtype == bool else (fb != 0))[0])
        self.list_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, np.array([1 if i in self.frozen else 0 for i in range(self.N)])), 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L0 = np.full((N, n + 1), np.nan, dtype=np.float64)
        B0 = np.full((N, n + 1), np.nan)
        L0[:, 0] = llr_ch
        paths = [{"pm": 0.0, "L": L0, "B": B0, "u": np.zeros(N, dtype=int)}]

        for phase_i in range(N):
            l = bit_reversed(phase_i, n)
            new_paths = []
            for path in paths:
                _update_llrs_path(path["L"], path["B"], l, n)
                llr = path["L"][l, n]
                if l in self.frozen:
                    p2 = {
                        "pm": _pm_update(path["pm"], llr, 0),
                        "L": path["L"].copy(),
                        "B": path["B"].copy(),
                        "u": path["u"].copy(),
                    }
                    p2["B"][l, n] = 0
                    p2["u"][l] = 0
                    _update_bits(p2["B"], l, n)
                    new_paths.append(p2)
                else:
                    for u in (0, 1):
                        p2 = {
                            "pm": _pm_update(path["pm"], llr, u),
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                            "u": path["u"].copy(),
                        }
                        p2["B"][l, n] = u
                        p2["u"][l] = u
                        _update_bits(p2["B"], l, n)
                        new_paths.append(p2)
            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            info_idx = [i for i in range(N) if i not in self.frozen]
            valid = [p for p in paths if crc_check(p["u"][info_idx], self.crc_length)]
            best = min(valid if valid else paths, key=lambda p: p["pm"])
        else:
            best = paths[0]

        return best["u"], best["pm"]
