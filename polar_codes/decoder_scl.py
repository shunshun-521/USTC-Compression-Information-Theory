"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversed
from decoder_sc import (
    f_operation,
    g_operation,
    _active_llr_level,
    _active_bit_level,
)


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, n_bits):
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (n_bits - 1)
        for _ in range(n_bits):
            if n_bits == 8:
                msb = reg & 0x80
                reg = (reg << 1) & 0xFF
                if msb:
                    reg ^= poly
            else:
                msb = reg & 0x8000
                reg = (reg << 1) & 0xFFFF
                if msb:
                    reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    n_bits = crc_length
    remainder = _crc_remainder(info_bits, poly, n_bits)
    crc_bits = np.array(
        [(remainder >> (n_bits - 1 - i)) & 1 for i in range(n_bits)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=int)
    payload = bits[:-crc_length]
    expected = crc_encode(payload, crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


class _Path:
    __slots__ = ('L', 'B', 'pm', 'u_hat')

    def __init__(self, N, n, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = info_indices

    def _update_llrs(self, path, phi):
        for s in range(self.n - _active_llr_level(phi, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size >> 1
            for j in range(phi, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s],
                        path.L[j, s],
                        path.B[j - branch_size, s + 1],
                    )

    def _update_bits(self, path, phi):
        if phi < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(phi, self.n), -1):
            block_size = 1 << s
            branch_size = block_size >> 1
            for j in range(phi, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = path.B[j, s] ^ path.B[j - branch_size, s]
                    path.B[j, s - 1] = path.B[j, s]

    def _path_metric_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def _clone_path(self, src):
        dst = _Path(self.N, self.n, np.zeros(self.N))
        dst.L = src.L.copy()
        dst.B = src.B.copy()
        dst.pm = src.pm
        dst.u_hat = src.u_hat.copy()
        return dst

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for i in range(self.N):
            phi = bit_reversed(i, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, phi)
                llr_bit = path.L[phi, self.n]

                if phi in self.frozen_set:
                    new_path = self._clone_path(path)
                    new_path.pm += self._path_metric_penalty(llr_bit, 0)
                    new_path.u_hat[phi] = 0
                    new_path.B[phi, self.n] = 0
                    self._update_bits(new_path, phi)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = self._clone_path(path)
                        new_path.pm += self._path_metric_penalty(llr_bit, bit)
                        new_path.u_hat[phi] = bit
                        new_path.B[phi, self.n] = bit
                        self._update_bits(new_path, phi)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                if self.info_indices is not None:
                    bits = p.u_hat[self.info_indices]
                else:
                    bits = p.u_hat
                if crc_check(bits, self.crc_length):
                    valid.append(p)
            best = min(valid if valid else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
