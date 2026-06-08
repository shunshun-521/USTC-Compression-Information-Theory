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
    _update_bits,
    _update_llrs,
    f_operation,
    g_operation,
)
from encoder import bit_reversal_permutation


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for bit in info_bits:
        reg ^= (int(bit) << (crc_length - 1))
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    return np.array_equal(bits[-crc_length:], crc_encode(bits[:-crc_length], crc_length)[-crc_length:])


class PathState:
    __slots__ = ('pm', 'L', 'B', 'u_hat')

    def __init__(self, N, n, llr_init):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.u_hat = np.zeros(N, dtype=int)
        self.L[:, 0] = llr_init

    def copy(self):
        p = PathState(len(self.u_hat), self.L.shape[1] - 1, self.L[:, 0])
        p.pm = self.pm
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.u_hat = self.u_hat.copy()
        return p


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.rev = bit_reversal_permutation(N)
        self.decode_order = [_bit_reversed_index(i, self.n) for i in range(N)]

    def _path_llr(self, path, l):
        _update_llrs(path.L, path.B, l, self.n, self.N)
        return path.L[l, self.n]

    def _path_update_bits(self, path, l, bit):
        path.B[l, self.n] = bit
        path.u_hat[l] = bit
        _update_bits(path.B, l, self.n, self.N)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_init = llr_ch[self.rev]
        paths = [PathState(self.N, self.n, llr_init)]

        for l in self.decode_order:
            candidates = []
            for path in paths:
                llr = self._path_llr(path, l)
                if self.frozen_bits[l]:
                    pm = path.pm + (0.0 if llr >= 0 else abs(llr))
                    new_path = path.copy()
                    new_path.pm = pm
                    self._path_update_bits(new_path, l, 0)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        pm = path.pm + (0.0 if (bit == 0 and llr >= 0) or (bit == 1 and llr < 0) else abs(llr))
                        new_path = path.copy()
                        new_path.pm = pm
                        self._path_update_bits(new_path, l, bit)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[:self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
