"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    _SCDState,
    _bitrev_scalar,
    _frozen_set_from_mask,
    channel_llr_to_decoder,
    precompute_sc_indices,
)
from encoder import _bitrev_indices

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc8_remainder(bits):
    reg = 0
    for b in bits:
        reg ^= int(b) << 7
        for _ in range(8):
            if reg & 0x80:
                reg = ((reg << 1) & 0xFF) ^ CRC8_POLY
            else:
                reg = (reg << 1) & 0xFF
    return reg


def _crc16_remainder(bits):
    reg = 0
    for b in bits:
        reg ^= int(b) << 15
        for _ in range(16):
            if reg & 0x8000:
                reg = ((reg << 1) & 0xFFFF) ^ CRC16_POLY
            else:
                reg = (reg << 1) & 0xFFFF
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        rem = _crc8_remainder(info_bits)
        crc_bits = np.array([(rem >> i) & 1 for i in range(7, -1, -1)], dtype=np.int8)
    elif crc_length == 16:
        rem = _crc16_remainder(info_bits)
        crc_bits = np.array([(rem >> i) & 1 for i in range(15, -1, -1)], dtype=np.int8)
    else:
        raise ValueError("crc_length must be 8 or 16")
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    payload = bits[:-crc_length]
    expected = crc_encode(payload, crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


class SCLDecoder:
    """SCL 译码器（路径复制 + 路径度量）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_set = _frozen_set_from_mask(frozen_bits)
        self.L = list_size
        self.crc_length = crc_length
        self.info_indices = np.array(sorted(set(range(N)) - self.frozen_set), dtype=int)

    def _pm_penalty(self, llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = channel_llr_to_decoder(llr_ch)
        paths = [_SCDState(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = _bitrev_scalar(i, self.n)
            new_paths = []
            for path in paths:
                path.update_llrs(l)
                llr = path.L[l, self.n]
                if l in self.frozen_set:
                    path.pm += self._pm_penalty(llr, 0)
                    path.B[l, self.n] = 0
                    path.update_bits(l)
                    new_paths.append(path)
                else:
                    for u in (0, 1):
                        child = path.copy()
                        child.pm += self._pm_penalty(llr, u)
                        child.B[l, self.n] = u
                        child.update_bits(l)
                        new_paths.append(child)
            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.L]

        best = None
        best_pm = None
        for path in paths:
            u_hat = path.B[:, self.n].astype(int)
            if self.crc_length > 0:
                info_bits = u_hat[self.info_indices]
                if not crc_check(info_bits, self.crc_length):
                    continue
            if best_pm is None or path.pm < best_pm:
                best_pm = path.pm
                best = path

        if best is None:
            best = min(paths, key=lambda p: p.pm)
        u_hat = best.B[:, self.n].astype(int)
        return u_hat, float(best.pm)
