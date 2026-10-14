"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    bit_reversal_permutation,
    f_operation,
    g_operation,
    _active_llr_level,
    _active_bit_level,
    _update_llrs,
    _update_bits,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """CRC 校验位（MSB 先行）"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in info_bits:
        msb = (reg >> (crc_length - 1)) & 1
        reg = ((reg << 1) & mask) ^ int(bit)
        if msb:
            reg ^= poly
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits.astype(int), crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in bits:
        msb = (reg >> (crc_length - 1)) & 1
        reg = ((reg << 1) & mask) ^ int(bit)
        if msb:
            reg ^= poly
    return reg == 0


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)
        self.frozen_set = set(np.where(self.frozen_bits.astype(bool))[0])

    def _new_path(self, llr_ch):
        L = np.zeros((self.N, self.n + 1), dtype=np.float64)
        B = np.zeros((self.N, self.n + 1), dtype=int)
        L[:, 0] = llr_ch
        return {"pm": 0.0, "L": L, "B": B, "u_hat": np.zeros(self.N, dtype=int)}

    def _pm_add(self, pm, llr, u_bit):
        hard = 0 if llr >= 0 else 1
        if u_bit != hard:
            pm += abs(llr)
        return pm

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [self._new_path(llr_ch)]

        for i in range(self.N):
            l = self.br[i]
            candidates = []
            for path in paths:
                _update_llrs(path["L"], path["B"], l, self.n)
                llr = path["L"][l, self.n]
                if l in self.frozen_set:
                    pm = self._pm_add(path["pm"], llr, 0)
                    candidates.append((pm, path, 0))
                else:
                    for u_bit in (0, 1):
                        pm = self._pm_add(path["pm"], llr, u_bit)
                        candidates.append((pm, path, u_bit))

            candidates.sort(key=lambda x: x[0])
            new_paths = []
            for pm, parent, u_bit in candidates[: self.list_size]:
                child = self._new_path(llr_ch)
                child["pm"] = pm
                child["u_hat"] = parent["u_hat"].copy()
                child["u_hat"][l] = u_bit
                child["L"] = parent["L"].copy()
                child["B"] = parent["B"].copy()
                child["B"][l, self.n] = u_bit
                _update_bits(child["B"], l, self.n, self.N)
                new_paths.append(child)
            paths = new_paths

        paths.sort(key=lambda p: p["pm"])
        if self.crc_length > 0:
            for p in paths:
                if crc_check(p["u_hat"], self.crc_length):
                    return p["u_hat"], p["pm"]
        best = paths[0]
        return best["u_hat"], best["pm"]
