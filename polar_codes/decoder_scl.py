"""
极化码 SCL（串行抵消列表）译码器，支持 CRC 辅助 CA-SCL
"""
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    _upper_llr,
    _lower_llr,
    sc_decode_recursive,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int).ravel()
    if len(bits) < crc_length:
        return False
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class SCLDecoder:
    """SCL 译码器：多路径复制 P/B 数组"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _init_path(self, llr_ch):
        L = np.zeros((self.N, self.n + 1), dtype=np.float64)
        B = np.zeros((self.N, self.n + 1), dtype=np.int8)
        L[:, 0] = llr_ch
        return {"pm": 0.0, "L": L, "B": B, "u": np.zeros(self.N, dtype=int)}

    def _llr_at_phi(self, path, phi):
        l = _bit_reversed(phi, self.n)
        L, B = path["L"], path["B"]
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = _lower_llr(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )
        return float(L[l, self.n])

    def _update_bits(self, path, phi):
        l = _bit_reversed(phi, self.n)
        B = path["B"]
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [self._init_path(llr_ch)]

        for phi in range(self.N):
            candidates = []
            for path in paths:
                llr_bit = self._llr_at_phi(path, phi)
                if self.frozen_bits[phi]:
                    u = 0
                    pm = path["pm"] + (abs(llr_bit) if llr_bit < 0 else 0.0)
                    new_path = {
                        "pm": pm,
                        "L": path["L"].copy(),
                        "B": path["B"].copy(),
                        "u": path["u"].copy(),
                    }
                    l = _bit_reversed(phi, self.n)
                    new_path["B"][l, self.n] = 0
                    new_path["u"][phi] = 0
                    self._update_bits(new_path, phi)
                    candidates.append(new_path)
                else:
                    for u in (0, 1):
                        pm = path["pm"] + (0.0 if (u == 0 and llr_bit >= 0) or (u == 1 and llr_bit < 0) else abs(llr_bit))
                        new_path = {
                            "pm": pm,
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                            "u": path["u"].copy(),
                        }
                        l = _bit_reversed(phi, self.n)
                        new_path["B"][l, self.n] = u
                        new_path["u"][phi] = u
                        self._update_bits(new_path, phi)
                        candidates.append(new_path)
            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            ok = [p for p in paths if crc_check(p["u"][self.info_indices], self.crc_length)]
            if ok:
                paths = ok

        best = min(paths, key=lambda p: p["pm"])
        return best["u"].copy(), best["pm"]
