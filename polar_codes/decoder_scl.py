"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed_index,
    _lower_llr_exact,
    _to_natural_order,
    _upper_llr_exact,
    sc_decode,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_register(crc_length):
    if crc_length == 8:
        return CRC8_POLY, 8
    if crc_length == 16:
        return CRC16_POLY, 16
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后"""
    poly, r = _crc_register(crc_length)
    reg = 0
    bits = np.asarray(info_bits, dtype=int).tolist()
    for b in bits:
        fb = ((reg >> (r - 1)) & 1) ^ b
        reg = ((reg << 1) & ((1 << r) - 1)) ^ (poly if fb else 0)
    for _ in range(r):
        fb = (reg >> (r - 1)) & 1
        reg = ((reg << 1) & ((1 << r) - 1)) ^ (poly if fb else 0)
    crc_bits = [(reg >> (r - 1 - i)) & 1 for i in range(r)]
    return np.array(bits + crc_bits, dtype=int)


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    poly, r = _crc_register(crc_length)
    reg = 0
    for b in np.asarray(bits, dtype=int):
        fb = ((reg >> (r - 1)) & 1) ^ int(b)
        reg = ((reg << 1) & ((1 << r) - 1)) ^ (poly if fb else 0)
    return reg == 0


class _Path:
    __slots__ = ("L", "B", "pm", "u_phase")

    def __init__(self, N, n, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_phase = {}


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时浅拷贝数组）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_phase = {
            _bit_reversed_index(i, self.n) for i in np.where(self.frozen_bits > 0)[0]
        }
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def _pm_update(self, pm, llr, u):
        v = 0 if llr >= 0 else 1
        if u != v:
            pm += abs(llr)
        return pm

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            u = sc_decode(llr_ch, self.frozen_bits)
            return u, 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = _bit_reversed_index(i, self.n)
            new_paths = []
            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, self.n]
                if l in self.frozen_phase:
                    pm = self._pm_update(path.pm, llr, 0)
                    child = self._fork(path)
                    child.pm = pm
                    child.u_phase[l] = 0
                    child.B[l, self.n] = 0
                    self._update_bits(child, l)
                    new_paths.append(child)
                else:
                    for u in (0, 1):
                        pm = self._pm_update(path.pm, llr, u)
                        child = self._fork(path)
                        child.pm = pm
                        child.u_phase[l] = u
                        child.B[l, self.n] = u
                        self._update_bits(child, l)
                        new_paths.append(child)
            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        best = paths[0]
        if self.crc_length > 0:
            valid = []
            for p in paths:
                u_nat = self._path_to_natural(p)
                payload = u_nat[self.info_indices]
                if crc_check(payload, self.crc_length):
                    valid.append(p)
            if valid:
                best = min(valid, key=lambda p: p.pm)

        return self._path_to_natural(best), best.pm

    def _fork(self, path):
        child = object.__new__(_Path)
        child.L = path.L.copy()
        child.B = path.B.copy()
        child.pm = path.pm
        child.u_phase = dict(path.u_phase)
        return child

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block = 2 ** (s + 1)
            half = block // 2
            for j in range(l, self.N, block):
                if j % block < half:
                    path.L[j, s + 1] = _upper_llr_exact(path.L[j, s], path.L[j + half, s])
                else:
                    path.L[j, s + 1] = _lower_llr_exact(
                        path.L[j, s], path.L[j - half, s], int(path.B[j - half, s + 1])
                    )

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block = 2 ** s
            half = block // 2
            for j in range(l, -1, -block):
                if j % block >= half:
                    path.B[j - half, s - 1] = int(path.B[j, s]) ^ int(path.B[j - half, s])
                    path.B[j, s - 1] = path.B[j, s]

    def _path_to_natural(self, path):
        u_phase = np.zeros(self.N, dtype=int)
        for l, b in path.u_phase.items():
            u_phase[l] = b
        return _to_natural_order(u_phase, self.N)
