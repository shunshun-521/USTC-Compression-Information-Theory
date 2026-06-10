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
    f_operation,
    g_operation,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    for _ in range(crc_length):
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _update_llrs(self, L, B, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )

    def _update_bits(self, B, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n, L = self.N, self.n, self.list_size

        paths = [{
            "pm": 0.0,
            "L": np.full((N, n + 1), np.nan, dtype=np.float64),
            "B": np.zeros((N, n + 1), dtype=int),
            "u": np.zeros(N, dtype=int),
            "active": True,
        }]
        paths[0]["L"][:, 0] = llr_ch

        for phi in range(N):
            l = _bit_reversed(phi, n)
            candidates = []

            for path in paths:
                if not path["active"]:
                    continue
                self._update_llrs(path["L"], path["B"], l)
                llr_val = path["L"][l, n]

                if l in self.frozen_set:
                    penalty = 0.0 if llr_val >= 0 else abs(llr_val)
                    nc = {
                        "pm": path["pm"] + penalty,
                        "L": path["L"].copy(),
                        "B": path["B"].copy(),
                        "u": path["u"].copy(),
                        "active": True,
                    }
                    nc["u"][l] = 0
                    nc["B"][l, n] = 0
                    candidates.append(nc)
                else:
                    for bit in (0, 1):
                        penalty = (
                            0.0 if bit == (0 if llr_val >= 0 else 1) else abs(llr_val)
                        )
                        nc = {
                            "pm": path["pm"] + penalty,
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                            "u": path["u"].copy(),
                            "active": True,
                        }
                        nc["u"][l] = bit
                        nc["B"][l, n] = bit
                        candidates.append(nc)

            candidates.sort(key=lambda c: c["pm"])
            paths = candidates[:L]
            while len(paths) < L:
                paths.append({
                    "pm": np.inf,
                    "L": np.full((N, n + 1), np.nan),
                    "B": np.zeros((N, n + 1), dtype=int),
                    "u": np.zeros(N, dtype=int),
                    "active": False,
                })

            for path in paths:
                if path["active"]:
                    self._update_bits(path["B"], l)

        active = [p for p in paths if p["active"]]
        if self.crc_length > 0:
            valid = [
                p for p in active
                if crc_check(p["u"][self.info_indices], self.crc_length)
            ]
            best = min(valid if valid else active, key=lambda p: p["pm"])
        else:
            best = min(active, key=lambda p: p["pm"])

        return best["u"], best["pm"]
