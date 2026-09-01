"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _update_bits,
    _update_llrs,
    f_operation,
    g_operation,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _poly_div_crc(info_bits, crc_length):
    """GF(2) 多项式除法求 CRC 余式"""
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    poly_full = (1 << crc_length) | poly
    msg = list(map(int, info_bits)) + [0] * crc_length
    for i in range(len(info_bits)):
        if msg[i]:
            for j in range(crc_length + 1):
                if (poly_full >> (crc_length - j)) & 1:
                    msg[i + j] ^= 1
    return np.array(msg[-crc_length:], dtype=int)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    crc_bits = _poly_div_crc(info_bits, crc_length)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int).ravel()
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    poly_full = (1 << crc_length) | poly
    msg = list(map(int, bits))
    n = len(bits) - crc_length
    for i in range(n):
        if msg[i]:
            for j in range(crc_length + 1):
                if (poly_full >> (crc_length - j)) & 1:
                    msg[i + j] ^= 1
    return all(v == 0 for v in msg[-crc_length:])


class Path:
    __slots__ = ('L', 'B', 'pm', 'active')

    def __init__(self, N, n, llr_ch):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.active = True


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 LLR/比特数组）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_set = np.where(~self.frozen_bits)[0]

    def _llr_to_bit(self, llr_val):
        return 0 if llr_val >= 0 else 1

    def _pm_update(self, pm, llr_val, bit):
        expected = self._llr_to_bit(llr_val)
        if bit != expected:
            pm += abs(llr_val)
        return pm

    def _advance_path(self, path, l):
        _update_llrs(path.L, path.B, l, self.n, self.N)
        return path.L[l, self.n]

    def _set_bit_and_propagate(self, path, l, bit):
        path.B[l, self.n] = bit
        _update_bits(path.B, l, self.n, self.N)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        rev_idx = np.array([_bit_reversed(i, self.n) for i in range(self.N)], dtype=int)
        llr_proc = llr_ch[rev_idx]

        paths = [Path(self.N, self.n, llr_proc.copy())]

        for phi in range(self.N):
            l = _bit_reversed(phi, self.n)
            candidates = []

            for path in paths:
                if not path.active:
                    continue
                llr_val = self._advance_path(path, l)

                if l in self.frozen_set:
                    pm = self._pm_update(path.pm, llr_val, 0)
                    new_path = Path(self.N, self.n, path.L[:, 0].copy())
                    new_path.L = path.L.copy()
                    new_path.B = path.B.copy()
                    new_path.pm = pm
                    self._set_bit_and_propagate(new_path, l, 0)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        pm = self._pm_update(path.pm, llr_val, bit)
                        new_path = Path(self.N, self.n, path.L[:, 0].copy())
                        new_path.L = path.L.copy()
                        new_path.B = path.B.copy()
                        new_path.pm = pm
                        self._set_bit_and_propagate(new_path, l, bit)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        best = None
        if self.crc_length > 0:
            for path in paths:
                u_hat = path.B[:, self.n].astype(int)
                info_bits = u_hat[self.info_set]
                if crc_check(info_bits, self.crc_length):
                    if best is None or path.pm < best.pm:
                        best = path
        if best is None:
            best = min(paths, key=lambda p: p.pm)

        return best.B[:, self.n].astype(int), best.pm
