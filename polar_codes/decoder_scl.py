"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math
from decoder_sc import (
    _prepare_llr, _upper_llr, _lower_llr,
    _active_llr_level, _active_bit_level, _bit_reversed,
)
from encoder import bit_reversal_permutation

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg = ((reg << 1) | int(bit)) & ((1 << (crc_length + 1)) - 1)
        if reg & (1 << crc_length):
            reg ^= poly
    for _ in range(crc_length):
        reg = (reg << 1) & ((1 << (crc_length + 1)) - 1)
        if reg & (1 << crc_length):
            reg ^= poly
    return reg & ((1 << crc_length) - 1)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class _Path:
    __slots__ = ('L', 'B', 'pm', 'u_hat')

    def __init__(self, N, n):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.float64)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)

    def copy(self):
        p = _Path(self.L.shape[0], self.L.shape[1] - 1)
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.pm = self.pm
        p.u_hat = self.u_hat.copy()
        return p


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _update_llrs(self, paths, l):
        for path in paths:
            for s in range(self.n - _active_llr_level(l, self.n), self.n):
                block_size = 2 ** (s + 1)
                branch_size = block_size // 2
                for j in range(l, self.N, block_size):
                    if j % block_size < branch_size:
                        path.L[j, s + 1] = _upper_llr(path.L[j, s], path.L[j + branch_size, s])
                    else:
                        path.L[j, s + 1] = _lower_llr(
                            path.L[j, s],
                            path.L[j - branch_size, s],
                            int(path.B[j - branch_size, s + 1]),
                        )

    def _update_bits(self, paths, l):
        if l < self.N / 2:
            return
        for path in paths:
            for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        path.B[j - branch_size, s - 1] = (
                            int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                        )
                        path.B[j, s - 1] = path.B[j, s]

    @staticmethod
    def _pm_penalty(llr_val, u_bit):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr_val)

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)。"""
        llr_ch = _prepare_llr(llr_ch)
        paths = [_Path(self.N, self.n)]
        paths[0].L[:, 0] = llr_ch

        for phi in range(self.N):
            l = _bit_reversed(phi, self.n)
            self._update_llrs(paths, l)
            llr_val = paths[0].L[l, self.n]

            if l in self.frozen_set:
                new_paths = []
                for path in paths:
                    np_path = path.copy()
                    np_path.u_hat[l] = 0
                    np_path.B[l, self.n] = 0
                    np_path.pm += self._pm_penalty(path.L[l, self.n], 0)
                    new_paths.append(np_path)
                paths = new_paths
            else:
                new_paths = []
                for path in paths:
                    lv = path.L[l, self.n]
                    for u_bit in (0, 1):
                        np_path = path.copy()
                        np_path.u_hat[l] = u_bit
                        np_path.B[l, self.n] = u_bit
                        np_path.pm += self._pm_penalty(lv, u_bit)
                        new_paths.append(np_path)
                new_paths.sort(key=lambda p: p.pm)
                paths = new_paths[:self.list_size]

            self._update_bits(paths, l)

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat[self.info_indices], self.crc_length)]
            best = min(valid if valid else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
