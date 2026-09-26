"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import _bit_reversed, f_operation, g_operation
from encoder import bit_reversal_permutation

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_run(data_bits, poly, crc_length):
    mask = (1 << crc_length) - 1
    reg = 0
    for b in data_bits:
        fb = ((reg >> (crc_length - 1)) & 1) ^ (int(b) & 1)
        reg = (reg << 1) & mask
        if fb:
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    rem = _crc_run(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_run(bits, poly, crc_length) == 0


class SCLDecoder:
    """SCL 译码器（Lazy Copy）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _new_path(self, llr_ch):
        L = np.zeros((self.N, self.n + 1), dtype=np.float64)
        B = np.zeros((self.N, self.n + 1), dtype=np.int8)
        L[:, self.n] = llr_ch
        return {"pm": 0.0, "L": L, "B": B, "u_hat": np.zeros(self.N, dtype=np.int8)}

    def _copy_path(self, path):
        return {
            "pm": path["pm"],
            "L": path["L"],
            "B": path["B"],
            "u_hat": path["u_hat"].copy(),
            "lazy": True,
        }

    def _materialize(self, path):
        if path.get("lazy"):
            path["L"] = path["L"].copy()
            path["B"] = path["B"].copy()
            path["lazy"] = False

    def _update_llr(self, path, x):
        L, B = path["L"], path["B"]
        n = self.n
        for j in range(n - 1, -1, -1):
            s = 1 << (n - j)
            t = s // 2
            for i in range(x, self.N, s):
                if t > (i % s):
                    L[i, j] = f_operation(L[i, j + 1], L[i + t, j + 1])
                else:
                    L[i, j] = g_operation(L[i, j + 1], L[i - t, j + 1], B[i - t, j])

    def _update_bits(self, path, x):
        B = path["B"]
        n = self.n
        active = [x]
        for j in range(n):
            s = 1 << (n - j)
            t = s // 2
            nxt = []
            for i in active:
                if t <= (i % s):
                    B[i - t, j + 1] = (B[i, j] + B[i - t, j]) % 2
                    B[i, j + 1] = B[i, j]
                    nxt.extend([i, i - t])
            active = nxt

    def _penalty(self, llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        br = bit_reversal_permutation(self.N)
        llr_ch = llr_ch[br]

        paths = [self._new_path(llr_ch)]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            candidates = []
            for path in paths:
                self._materialize(path)
                self._update_llr(path, l)
                llr = path["L"][l, 0]
                if self.frozen_bits[l]:
                    path["pm"] += self._penalty(llr, 0)
                    path["u_hat"][l] = 0
                    path["B"][l, 0] = 0
                    self._update_bits(path, l)
                    candidates.append(path)
                else:
                    for u in (0, 1):
                        child = self._copy_path(path)
                        self._materialize(child)
                        child["pm"] += self._penalty(llr, u)
                        child["u_hat"][l] = u
                        child["B"][l, 0] = u
                        self._update_bits(child, l)
                        candidates.append(child)
            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p["u_hat"][self.info_indices], self.crc_length)
            ]
            if valid:
                paths = valid
        best = min(paths, key=lambda p: p["pm"])
        return best["u_hat"].astype(int), best["pm"]
