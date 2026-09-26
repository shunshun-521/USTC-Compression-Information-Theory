"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    _SCDEngine,
    _channel_to_core_llr,
    bit_reversed_index,
    active_llr_level,
    active_bit_level,
    f_operation,
    g_operation,
)


def crc_encode(info_bits, crc_length=8):
    """CRC-8 (0x07) 或 CRC-16 (0x8005)"""
    info_bits = np.asarray(info_bits, dtype=np.uint8)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array([(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int)
    return np.concatenate([info_bits.astype(int), crc_bits])


def crc_check(bits, crc_length=8):
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=int)
    rec = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(rec, bits)


class SCLDecoder:
    """SCL 译码器（路径复制 + PM）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.L_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _pm_update(self, pm, llr, u):
        penalty = 0.0 if (u == 0 and llr >= 0) or (u == 1 and llr < 0) else abs(llr)
        return pm + penalty

    def decode(self, llr_ch):
        llr_core = _channel_to_core_llr(llr_ch)
        paths = [
            {
                "pm": 0.0,
                "L": np.zeros((self.N, self.n + 1), dtype=np.float64),
                "B": np.zeros((self.N, self.n + 1), dtype=np.int8),
                "u": np.zeros(self.N, dtype=int),
            }
        ]
        paths[0]["L"][:, 0] = llr_core

        for phase in range(self.N):
            l_idx = bit_reversed_index(phase, self.n)
            candidates = []
            for p in paths:
                self._update_llrs_path(p, l_idx)
                llr = p["L"][l_idx, self.n]
                if l_idx in self.frozen_set:
                    u = 0
                    pm = self._pm_update(p["pm"], llr, u)
                    cand = self._clone_path(p)
                    cand["pm"] = pm
                    cand["u"][l_idx] = u
                    cand["B"][l_idx, self.n] = 0
                    self._update_bits_path(cand, l_idx)
                    candidates.append(cand)
                else:
                    for u in (0, 1):
                        pm = self._pm_update(p["pm"], llr, u)
                        cand = self._clone_path(p)
                        cand["pm"] = pm
                        cand["u"][l_idx] = u
                        cand["B"][l_idx, self.n] = u
                        self._update_bits_path(cand, l_idx)
                        candidates.append(cand)
            candidates.sort(key=lambda x: x["pm"])
            paths = candidates[: self.L_size]

        best = paths[0]
        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p["u"][self.info_indices], self.crc_length)]
            if valid:
                best = min(valid, key=lambda x: x["pm"])
        return best["u"], best["pm"]

    @staticmethod
    def _clone_path(p):
        return {
            "pm": p["pm"],
            "L": p["L"].copy(),
            "B": p["B"].copy(),
            "u": p["u"].copy(),
        }

    def _update_llrs_path(self, p, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    p["L"][j, s + 1] = float(
                        f_operation(
                            np.array([p["L"][j, s]]),
                            np.array([p["L"][j + branch_size, s]]),
                        )[0]
                    )
                else:
                    p["L"][j, s + 1] = float(
                        g_operation(
                            np.array([p["L"][j - branch_size, s]]),
                            np.array([p["L"][j, s]]),
                            np.array([p["B"][j - branch_size, s + 1]]),
                        )[0]
                    )

    def _update_bits_path(self, p, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    p["B"][j - branch_size, s - 1] = int(p["B"][j, s]) ^ int(
                        p["B"][j - branch_size, s]
                    )
                    p["B"][j, s - 1] = p["B"][j, s]
