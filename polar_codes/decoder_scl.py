"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    _upper_llr,
    _lower_llr,
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
    """计算 CRC 并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    reg = 0
    for b in info_bits:
        msb = ((reg >> (crc_length - 1)) & 1) ^ int(b)
        reg = (reg << 1) & mask
        if msb:
            reg ^= poly
    crc_bits = [(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)]
    return np.concatenate([info_bits, np.array(crc_bits, dtype=np.int8)])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=np.int8)
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


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat", "parent", "fork_layer")

    def __init__(self, N, n, llr_ch, parent=None):
        self.L = parent.L.copy() if parent is not None else np.zeros((N, n + 1), dtype=np.float64)
        self.B = parent.B.copy() if parent is not None else np.zeros((N, n + 1), dtype=np.int8)
        if parent is None:
            rev = np.array([_bit_reversed(i, n) for i in range(N)], dtype=int)
            self.L[:, 0] = llr_ch[rev]
        self.pm = parent.pm if parent is not None else 0.0
        self.u_hat = parent.u_hat.copy() if parent is not None else np.zeros(N, dtype=int)
        self.parent = parent
        self.fork_layer = None


class SCLDecoder:
    """SCL 译码器（路径复制在比特判决时进行）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _path_metric_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            candidates = []

            for p in paths:
                _update_llrs(p.L, p.B, l, self.n)
                llr = p.L[l, self.n]

                if self.frozen_bits[l]:
                    bit = 0
                    new_pm = p.pm + self._path_metric_penalty(llr, bit)
                    child = _Path(self.N, self.n, llr_ch, parent=p)
                    child.pm = new_pm
                    child.u_hat[l] = bit
                    child.B[l, self.n] = bit
                    _update_bits(child.B, l, self.n, self.N)
                    candidates.append(child)
                else:
                    for bit in (0, 1):
                        new_pm = p.pm + self._path_metric_penalty(llr, bit)
                        child = _Path(self.N, self.n, llr_ch, parent=p)
                        child.pm = new_pm
                        child.u_hat[l] = bit
                        child.B[l, self.n] = bit
                        _update_bits(child.B, l, self.n, self.N)
                        candidates.append(child)

            candidates.sort(key=lambda x: x.pm)
            paths = candidates[: self.list_size]

        best = paths[0]
        if self.crc_length > 0:
            valid = [p for p in paths if self._crc_ok(p.u_hat)]
            if valid:
                best = min(valid, key=lambda x: x.pm)

        return best.u_hat.astype(int), float(best.pm)

    def _crc_ok(self, u_hat):
        info_bits = u_hat[self.info_indices]
        if len(info_bits) < self.crc_length:
            return False
        return crc_check(info_bits, self.crc_length)
