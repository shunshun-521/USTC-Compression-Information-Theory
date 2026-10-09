"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    bit_reversed,
    sc_update_llrs,
    sc_update_bits,
)


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    mask = (1 << crc_length) - 1
    reg = 0
    for b in bits:
        fb = (reg >> (crc_length - 1)) & 1
        reg = (reg << 1) & mask
        if fb ^ int(b):
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    for _ in range(crc_length):
        fb = (rem >> (crc_length - 1)) & 1
        rem = (rem << 1) & ((1 << crc_length) - 1)
        if fb:
            rem ^= poly
    crc_bits = np.array([(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int).ravel()
    if len(bits) < crc_length:
        return False
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_remainder(bits, poly, crc_length)
    return rem == 0


def _pm_penalty(llr, u):
    u_hard = 0 if llr >= 0 else 1
    return 0.0 if u == u_hard else abs(llr)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 L/B）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, min_sum=True):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen = set(np.where(np.asarray(frozen_bits, dtype=bool))[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.min_sum = min_sum
        self.info_positions = np.where(~np.asarray(frozen_bits, dtype=bool))[0]

    def _new_path(self, llr_ch):
        L = np.zeros((self.N, self.n + 1), dtype=np.float64)
        B = np.zeros((self.N, self.n + 1), dtype=np.int8)
        L[:, 0] = llr_ch
        return {"L": L, "B": B, "pm": 0.0, "u": np.zeros(self.N, dtype=int)}

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [self._new_path(llr_ch)]

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            candidates = []

            for path in paths:
                sc_update_llrs(path["L"], path["B"], l, self.n, self.N, self.min_sum)
                llr = path["L"][l, self.n]

                if l in self.frozen:
                    pen = _pm_penalty(llr, 0)
                    path["pm"] += pen
                    path["B"][l, self.n] = 0
                    path["u"][l] = 0
                    sc_update_bits(path["B"], l, self.n, self.N)
                    candidates.append(path)
                else:
                    for u in (0, 1):
                        child = {
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                            "pm": path["pm"] + _pm_penalty(llr, u),
                            "u": path["u"].copy(),
                        }
                        child["B"][l, self.n] = u
                        child["u"][l] = u
                        sc_update_bits(child["B"], l, self.n, self.N)
                        candidates.append(child)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        best_crc = None
        best_pm = None
        for path in paths:
            u_hat = path["u"]
            if self.crc_length > 0:
                info_bits = u_hat[self.info_positions]
                if crc_check(info_bits, self.crc_length):
                    if best_crc is None or path["pm"] < best_crc["pm"]:
                        best_crc = path
            if best_pm is None or path["pm"] < best_pm["pm"]:
                best_pm = path

        chosen = best_crc if best_crc is not None else best_pm
        return chosen["u"].copy(), float(chosen["pm"])
