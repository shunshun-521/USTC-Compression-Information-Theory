"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import active_bit_level, active_llr_level, lower_llr, upper_llr
from encoder import bit_reversed


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & top:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class SCLDecoder:
    """SCL 译码器（基于 Permuted SCD）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length

    def _update_llrs(self, L, B, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = lower_llr(
                        L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                    )

    def _update_bits(self, B, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        paths = [{
            "pm": 0.0,
            "L": np.full((self.N, self.n + 1), np.nan, dtype=np.float64),
            "B": np.full((self.N, self.n + 1), np.nan),
            "u_hat": np.zeros(self.N, dtype=int),
        }]
        paths[0]["L"][:, 0] = llr_ch

        for phi in range(self.N):
            l = bit_reversed(phi, self.n)
            is_frozen = l in self.frozen_set
            candidates = []

            for path in paths:
                self._update_llrs(path["L"], path["B"], l)
                llr = path["L"][l, self.n]

                if is_frozen:
                    pm = path["pm"] + (0.0 if llr >= 0 else abs(llr))
                    new_path = {
                        "pm": pm,
                        "L": path["L"].copy(),
                        "B": path["B"].copy(),
                        "u_hat": path["u_hat"].copy(),
                    }
                    new_path["B"][l, self.n] = 0
                    new_path["u_hat"][l] = 0
                    self._update_bits(new_path["B"], l)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        pm = path["pm"] + (
                            0.0
                            if (bit == 0 and llr >= 0) or (bit == 1 and llr < 0)
                            else abs(llr)
                        )
                        new_path = {
                            "pm": pm,
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                            "u_hat": path["u_hat"].copy(),
                        }
                        new_path["B"][l, self.n] = bit
                        new_path["u_hat"][l] = bit
                        self._update_bits(new_path["B"], l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        best_idx = 0
        if self.crc_length > 0:
            valid = [
                i for i, p in enumerate(paths)
                if crc_check(p["u_hat"][~self.frozen_bits], self.crc_length)
            ]
            if valid:
                best_idx = min(valid, key=lambda i: paths[i]["pm"])

        return paths[best_idx]["u_hat"], paths[best_idx]["pm"]
