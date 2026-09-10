"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL），基于 Permuted SCD
"""
import math

import numpy as np

from polar_decoder_core import bit_reversed, update_bits, update_llrs
from decoder_sc import sc_decode


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _bits_to_bytes(bits):
    bits = np.asarray(bits, dtype=np.int8).ravel()
    if len(bits) == 0:
        return bytes()
    pad = (-len(bits)) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.int8)])
    out = bytearray()
    for i in range(0, len(bits), 8):
        val = 0
        for j in range(8):
            val = (val << 1) | int(bits[i + j])
        out.append(val)
    return bytes(out)


def _crc8_byte(data, poly=0x07, init=0):
    crc = init
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ poly) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后（CRC-8: 0x07）"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length != 8:
        raise NotImplementedError('当前仅实现 CRC-8')
    crc_val = _crc8_byte(_bits_to_bytes(info_bits), CRC8_POLY)
    crc_bits = np.array([(crc_val >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC（对完整比特序列重新计算，余数应为 0）"""
    if crc_length <= 0:
        return True
    bits = np.asarray(bits, dtype=np.int8)
    if crc_length != 8:
        raise NotImplementedError('当前仅实现 CRC-8')
    return _crc8_byte(_bits_to_bytes(bits), CRC8_POLY) == 0


def _pm_update(pm, llr, u):
    u_from_llr = 0 if llr >= 0 else 1
    if u != u_from_llr:
        pm += abs(llr)
    return pm


def _update_llrs_path(L, B, l, n, N):
    update_llrs(L, B, l, n, N)


def _update_bits_path(B, l, n, N):
    update_bits(B, l, n, N)


class Path:
    __slots__ = ('L', 'B', 'pm', 'u_hat')

    def __init__(self, N, n, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)

    def copy(self):
        p = Path(self.L.shape[0], self.L.shape[1] - 1, self.L[:, 0])
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.pm = self.pm
        p.u_hat = self.u_hat.copy()
        return p


class SCLDecoder:
    """SCL 译码器（Lazy Copy）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_set = set(np.where(np.asarray(frozen_bits).astype(bool))[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~np.asarray(frozen_bits).astype(bool))[0]

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode(llr_ch, [i in self.frozen_set for i in range(self.N)])
            return u_hat, 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [Path(self.N, self.n, llr_ch)]

        for phase in range(self.N):
            l = bit_reversed(phase, self.n)
            candidates = []

            for path in paths:
                _update_llrs_path(path.L, path.B, l, self.n, self.N)
                llr_bit = path.L[l, self.n]

                if l in self.frozen_set:
                    p = path.copy()
                    p.pm = _pm_update(p.pm, llr_bit, 0)
                    p.u_hat[l] = 0
                    p.B[l, self.n] = 0
                    _update_bits_path(p.B, l, self.n, self.N)
                    candidates.append(p)
                else:
                    for u in (0, 1):
                        p = path.copy()
                        p.pm = _pm_update(p.pm, llr_bit, u)
                        p.u_hat[l] = u
                        p.B[l, self.n] = u
                        _update_bits_path(p.B, l, self.n, self.N)
                        candidates.append(p)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        crc_ok = []
        for p in paths:
            if self.crc_length > 0:
                payload = p.u_hat[self.info_indices]
                if crc_check(payload, self.crc_length):
                    crc_ok.append(p)

        best = min(crc_ok, key=lambda p: p.pm) if crc_ok else paths[0]
        return best.u_hat.copy(), best.pm
