"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed_index,
    _permute_channel_llr,
    sc_decode,
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
    CRC-8: 0x07; CRC-16: 0x8005
    """
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    if crc_length == 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


def _f_ms(l1, l2):
    v1, v2 = float(l1), float(l2)
    if np.isnan(v1) and np.isnan(v2):
        return np.nan
    if np.isnan(v1):
        return v2
    if np.isnan(v2):
        return v1
    return np.sign(v1) * np.sign(v2) * min(abs(v1), abs(v2))


def _g_ms(l1, l2, b):
    v1, v2 = float(l1), float(l2)
    if np.isnan(v1) and np.isnan(v2):
        return np.nan
    if np.isnan(v1):
        return v2
    if np.isnan(v2):
        return v1
    return (v1 + v2) if b == 0 else (v1 - v2)


class _Path:
    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, llr, N, n):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.L[:, 0] = llr
        self.u_hat = np.zeros(N, dtype=int)

    def clone(self):
        child = _Path.__new__(_Path)
        child.pm = self.pm
        child.L = self.L.copy()
        child.B = self.B.copy()
        child.u_hat = self.u_hat.copy()
        return child


def _update_llrs(path, l, N, n):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size >> 1
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                path.L[j, s + 1] = _f_ms(path.L[j, s], path.L[j + branch_size, s])
            else:
                top = path.B[j - branch_size, s + 1]
                path.L[j, s + 1] = _g_ms(
                    path.L[j, s], path.L[j - branch_size, s], 0 if np.isnan(top) else int(top)
                )


def _update_bits(path, l, N, n):
    if l < N / 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size >> 1
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                path.B[j, s - 1] = path.B[j, s]


class SCLDecoder:
    """SCL 译码器（分层 LLR/比特存储，路径复制）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = max(1, int(list_size))
        self.crc_length = int(crc_length)
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    @staticmethod
    def _path_metric_penalty(llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr = _permute_channel_llr(np.asarray(llr_ch, dtype=np.float64), self.N)
        paths = [_Path(llr, self.N, self.n)]

        for phi in range(self.N):
            l = _bit_reversed_index(phi, self.n)
            new_paths = []

            for path in paths:
                _update_llrs(path, l, self.N, self.n)
                llr_bit = path.L[l, self.n]

                if l in self.frozen_set:
                    path.pm += self._path_metric_penalty(llr_bit, 0)
                    path.B[l, self.n] = 0
                    path.u_hat[l] = 0
                    _update_bits(path, l, self.N, self.n)
                    new_paths.append(path)
                else:
                    for bit in (0, 1):
                        child = path.clone()
                        child.pm += self._path_metric_penalty(llr_bit, bit)
                        child.B[l, self.n] = bit
                        child.u_hat[l] = bit
                        _update_bits(child, l, self.N, self.n)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            chosen = valid[0] if valid else paths[0]
        else:
            chosen = paths[0]

        return chosen.u_hat.copy(), chosen.pm
