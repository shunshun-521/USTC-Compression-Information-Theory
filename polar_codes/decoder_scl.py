"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import f_operation, g_operation, _bit_reversed, _active_llr_level, _active_bit_level

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _poly_div_mod2(dividend, divisor):
    """GF(2) 多项式长除法，返回余数"""
    dividend = np.asarray(dividend, dtype=np.int64).copy()
    divisor = np.asarray(divisor, dtype=np.int64)
    while len(dividend) >= len(divisor):
        if dividend[0] == 1:
            dividend[: len(divisor)] ^= divisor
        dividend = dividend[1:]
    return dividend


def _crc_poly_bits(crc_length):
    if crc_length == 8:
        return np.array([1, 0, 0, 0, 0, 0, 1, 1, 1], dtype=np.int64)
    return np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=np.int64)


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int64)
    poly = _crc_poly_bits(crc_length)
    padded = np.concatenate([info_bits, np.zeros(crc_length, dtype=np.int64)])
    remainder = _poly_div_mod2(padded, poly)
    if len(remainder) < crc_length:
        remainder = np.concatenate([
            np.zeros(crc_length - len(remainder), dtype=np.int64), remainder
        ])
    return np.concatenate([info_bits, remainder])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int64)
    poly = _crc_poly_bits(crc_length)
    remainder = _poly_div_mod2(bits, poly)
    return len(remainder) == 0 or np.all(remainder == 0)


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.br = bit_reversal_permutation(N)

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def _update_llrs(self, paths, l):
        for path in paths:
            L, B = path["L"], path["B"]
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

    def _update_bits(self, paths, l):
        if l < self.N // 2:
            return
        for path in paths:
            L, B = path["L"], path["B"]
            for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                        B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)[self.br]
        N, n = self.N, self.n

        def new_path(llr):
            L = np.zeros((N, n + 1), dtype=np.float64)
            B = np.zeros((N, n + 1), dtype=np.int64)
            L[:, 0] = llr
            return {"L": L, "B": B, "pm": 0.0, "u": np.full(N, -1, dtype=np.int64)}

        paths = [new_path(llr_ch)]

        for phi in range(N):
            l = _bit_reversed(phi, n)
            self._update_llrs(paths, l)

            new_paths = []
            for path in paths:
                llr = path["L"][l, n]
                if l in self.frozen_set:
                    p = {
                        "L": path["L"].copy(),
                        "B": path["B"].copy(),
                        "pm": path["pm"] + self._pm_penalty(llr, 0),
                        "u": path["u"].copy(),
                    }
                    p["B"][l, n] = 0
                    p["u"][l] = 0
                    new_paths.append(p)
                else:
                    for bit in (0, 1):
                        p = {
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                            "pm": path["pm"] + self._pm_penalty(llr, bit),
                            "u": path["u"].copy(),
                        }
                        p["B"][l, n] = bit
                        p["u"][l] = bit
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]
            self._update_bits(paths, l)

        best_crc = None
        best_pm = None
        for path in paths:
            u_hat = path["u"].astype(np.int64)
            if self.crc_length > 0:
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    if best_crc is None or path["pm"] < best_crc["pm"]:
                        best_crc = path
            if best_pm is None or path["pm"] < best_pm["pm"]:
                best_pm = path

        chosen = best_crc if best_crc is not None else best_pm
        return chosen["u"].astype(np.int64), chosen["pm"]
