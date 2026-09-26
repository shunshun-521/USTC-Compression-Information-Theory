"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import copy

import numpy as np

from decoder_sc import (
    _SCDCore,
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    lower_llr,
    sc_decode,
    upper_llr,
)
from encoder import bit_reversal_permutation


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.uint8)
    poly = _crc_poly(crc_length)
    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.uint8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=np.uint8)
    poly = _crc_poly(crc_length)
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


class SCLDecoder:
    """SCL 译码器（列表路径各自维护 SCD 状态）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_idx = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self._brp = bit_reversal_permutation(N)

    def _pm_penalty(self, llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def _step_path(self, core, i):
        l = _bit_reversed(i, self.n)
        core._update_llrs(l)
        return l, core.L[l, self.n]

    def _apply_bit(self, core, l, u_bit):
        core.B[l, self.n] = u_bit
        core._update_bits(l)

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode(llr_ch, self.frozen_bits.astype(int))
            return u_hat, 0.0

        llr = np.asarray(llr_ch, dtype=np.float64)[self._brp]
        paths = [{"core": _SCDCore(self.N, self.frozen_idx), "pm": 0.0}]
        paths[0]["core"].L[:, 0] = llr

        for i in range(self.N):
            expanded = []
            for path in paths:
                l, llr_bit = self._step_path(path["core"], i)
                if l in self.frozen_idx:
                    pm = path["pm"] + self._pm_penalty(llr_bit, 0)
                    new_core = copy.deepcopy(path["core"])
                    self._apply_bit(new_core, l, 0)
                    expanded.append({"core": new_core, "pm": pm})
                else:
                    for u in (0, 1):
                        pm = path["pm"] + self._pm_penalty(llr_bit, u)
                        new_core = copy.deepcopy(path["core"])
                        self._apply_bit(new_core, l, u)
                        expanded.append({"core": new_core, "pm": pm})
            expanded.sort(key=lambda p: p["pm"])
            paths = expanded[: self.list_size]

        paths.sort(key=lambda p: p["pm"])
        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p["core"].B[:, self.n].astype(int)[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            best = valid[0] if valid else paths[0]
        else:
            best = paths[0]

        u_hat = best["core"].B[:, self.n].astype(int)
        return u_hat, best["pm"]
