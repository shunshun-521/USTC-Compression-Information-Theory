"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    bit_reversed,
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
    """计算 CRC 并附加到信息比特末尾"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """校验 CRC"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat", "parent")

    def __init__(self, N, n, llr_ch, parent=None):
        if parent is None:
            self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
            self.B = np.zeros((N, n + 1), dtype=np.int8)
            self.L[:, 0] = llr_ch
            self.u_hat = np.zeros(N, dtype=np.int8)
            self.pm = 0.0
            self.parent = None
        else:
            self.L = parent.L.copy()
            self.B = parent.B.copy()
            self.u_hat = parent.u_hat.copy()
            self.pm = parent.pm
            self.parent = parent


class SCLDecoder:
    """SCL 译码器（Lazy Copy：分裂时共享 L，复制 B）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _path_llr_at(self, path, l):
        path.L = path.L.copy()
        _update_llrs(path.L, path.B, l, self.n)
        return path.L[l, self.n]

    def _path_apply_bit(self, path, l, bit, llr_val):
        path.B = path.B.copy()
        path.B[l, self.n] = bit
        path.u_hat[l] = bit
        _update_bits(path.B, l, self.n)
        hard = 0 if llr_val >= 0 else 1
        if bit != hard:
            path.pm += abs(llr_val)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            candidates = []

            for path in paths:
                llr_val = self._path_llr_at(path, l)
                if l in self.frozen_set:
                    p0 = _Path(self.N, self.n, llr_ch, parent=path)
                    self._path_apply_bit(p0, l, 0, llr_val)
                    candidates.append(p0)
                else:
                    for bit in (0, 1):
                        p = _Path(self.N, self.n, llr_ch, parent=path)
                        self._path_apply_bit(p, l, bit, llr_val)
                        candidates.append(p)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        # 选择最优路径（CRC 优先）
        if self.crc_length > 0:
            passed = []
            for p in paths:
                info_bits = p.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    passed.append(p)
            pool = passed if passed else paths
        else:
            pool = paths

        best = min(pool, key=lambda p: p.pm)
        return best.u_hat.astype(int), best.pm
