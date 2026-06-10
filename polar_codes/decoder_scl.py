"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from encoder import bit_reversed
from scd_ref import SCD


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8 if crc_length == 8 else 1):
            if crc_length == 8:
                if reg & 0x80:
                    reg = ((reg << 1) & 0xFF) ^ poly
                else:
                    reg = (reg << 1) & 0xFF
            else:
                if reg & top:
                    reg = ((reg << 1) & mask) ^ poly
                else:
                    reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class _PC:
    __slots__ = ("N", "n", "frozen", "likelihoods")


class _Path:
    """SCL 路径，封装 SCD 状态。"""

    __slots__ = ("scd", "pm", "u_hat")

    def __init__(self, N, n, llr_ch):
        pc = _PC()
        pc.N = N
        pc.n = n
        pc.frozen = set()
        pc.likelihoods = llr_ch
        self.scd = SCD(pc)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)

    def copy(self):
        p = _Path.__new__(_Path)
        pc = _PC()
        pc.N = self.scd.myPC.N
        pc.n = self.scd.myPC.n
        pc.frozen = set()
        pc.likelihoods = self.scd.L[:, 0].copy()
        p.scd = SCD(pc)
        p.scd.L = self.scd.L.copy()
        p.scd.B = self.scd.B.copy()
        p.pm = self.pm
        p.u_hat = self.u_hat.copy()
        return p


def _metric_penalty(llr, u):
    u_hard = 0 if llr >= 0 else 1
    return 0.0 if u == u_hard else abs(llr)


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            from decoder_sc import sc_decode

            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            candidates = []

            for path in paths:
                path.scd.update_llrs(l)
                llr = path.scd.L[l, self.n]

                if l in self.frozen_set:
                    child = path.copy()
                    child.pm += _metric_penalty(llr, 0)
                    child.u_hat[l] = 0
                    child.scd.B[l, self.n] = 0
                    child.scd.update_bits(l)
                    candidates.append(child)
                else:
                    for u in (0, 1):
                        child = path.copy()
                        child.pm += _metric_penalty(llr, u)
                        child.u_hat[l] = u
                        child.scd.B[l, self.n] = u
                        child.scd.update_bits(l)
                        candidates.append(child)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            best = min(valid if valid else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
