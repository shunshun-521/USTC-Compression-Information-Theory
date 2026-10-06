"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import math
import numpy as np

from decoder_sc import _SCDState, bit_reversed


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def _crc_update(reg, bit, poly, crc_length):
    mask = (1 << crc_length) - 1
    msb = (reg >> (crc_length - 1)) & 1
    reg = ((reg << 1) | int(bit)) & mask
    if msb ^ int(bit):
        reg ^= poly
    return reg & mask


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg = _crc_update(reg, bit, poly, crc_length)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in bits:
        reg = _crc_update(reg, bit, poly, crc_length)
    return reg == 0


class SCLDecoder:
    """SCL 译码器（路径级复制 L/B 数组）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])

    @staticmethod
    def _pm_penalty(llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n
        Lsize = self.list_size

        paths = [{"state": _SCDState(N, n, llr_ch.copy()), "pm": 0.0}]

        for i in range(N):
            l = bit_reversed(i, n)
            new_paths = []
            for path in paths:
                st = path["state"]
                st.update_llrs(l)
                llr = st.L[l, n]
                if l in self.frozen_set:
                    pm = path["pm"] + self._pm_penalty(llr, 0)
                    st2 = copy.deepcopy(st)
                    st2.B[l, n] = 0
                    st2.update_bits(l)
                    new_paths.append({"state": st2, "pm": pm})
                else:
                    for u in (0, 1):
                        st2 = copy.deepcopy(st)
                        pm = path["pm"] + self._pm_penalty(llr, u)
                        st2.B[l, n] = u
                        st2.update_bits(l)
                        new_paths.append({"state": st2, "pm": pm})
            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[:Lsize]

        paths.sort(key=lambda p: p["pm"])
        if self.crc_length > 0:
            info_idx = np.where(~self.frozen_bits)[0]
            for path in paths:
                u_hat = path["state"].B[:, n].astype(int)
                if crc_check(u_hat[info_idx], self.crc_length):
                    return u_hat.copy(), path["pm"]

        best = paths[0]
        u_hat = best["state"].B[:, n].astype(int)
        return u_hat.copy(), best["pm"]
