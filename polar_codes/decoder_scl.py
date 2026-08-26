"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import copy
import math

from decoder_sc import (
    _bit_reversed_index,
    _active_llr_level,
    _active_bit_level,
    _update_llrs,
    _update_bits,
    f_operation,
    g_operation,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8 (0x07) 或 CRC-16 (0x8005)
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg ^= (int(bit) << (crc_length - 1))
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits, expected)


class Path:
    def __init__(self, N, n):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int32)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时深拷贝 L/B）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        self.info_positions = np.where(self.frozen_bits == 0)[0]

    def _copy_path(self, path):
        new = Path(self.N, self.n)
        new.L = path.L.copy()
        new.B = path.B.copy()
        new.pm = path.pm
        new.u_hat = path.u_hat.copy()
        return new

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        root = Path(N, n)
        root.L[:, 0] = llr_ch
        active = [root]

        for phi in range(N):
            l = _bit_reversed_index(phi, n)
            candidates = []

            for path in active:
                _update_llrs(l, path.L, path.B, n)
                llr_bit = path.L[l, n]

                if l in self.frozen_set:
                    penalty = 0.0 if llr_bit >= 0 else abs(llr_bit)
                    new_path = self._copy_path(path)
                    new_path.pm += penalty
                    new_path.u_hat[l] = 0
                    new_path.B[l, n] = 0
                    _update_bits(l, new_path.B, n)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = self._copy_path(path)
                        if bit == 0:
                            penalty = 0.0 if llr_bit >= 0 else abs(llr_bit)
                        else:
                            penalty = 0.0 if llr_bit < 0 else abs(llr_bit)
                        new_path.pm += penalty
                        new_path.u_hat[l] = bit
                        new_path.B[l, n] = bit
                        _update_bits(l, new_path.B, n)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            active = candidates[:self.list_size]

        if self.crc_length > 0:
            crc_pass = [p for p in active if self._crc_path_check(p)]
            pool = crc_pass if crc_pass else active
        else:
            pool = active

        best = min(pool, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm

    def _crc_path_check(self, path):
        info_bits = path.u_hat[self.info_positions]
        return crc_check(info_bits, self.crc_length)
