"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from encoder import bit_reversal_int
from decoder_sc import (
    _active_llr_level,
    _active_bit_level,
    _boxplus,
    precompute_sc_indices,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_divide(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_divide(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> i) & 1 for i in range(crc_length - 1, -1, -1)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_divide(bits, poly, crc_length) == 0


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)

    def copy(self):
        p = _Path(self.L.shape[0], self.L.shape[1] - 1)
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.pm = self.pm
        p.u_hat = self.u_hat.copy()
        return p


def _lower_llr(l1, l2, bit):
    return l1 + l2 if bit == 0 else l1 - l2


def _update_llrs_path(L, B, l, n):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        N = L.shape[0]
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = _boxplus(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = _lower_llr(
                    L[j, s],
                    L[j - branch_size, s],
                    int(B[j - branch_size, s + 1]),
                )


def _update_bits_path(B, l, n):
    N = B.shape[0]
    if l < N / 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制数组）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        frozen_bits = np.asarray(frozen_bits)
        if frozen_bits.dtype == bool:
            self.frozen = set(np.where(frozen_bits)[0])
            self.info_mask = ~frozen_bits
        else:
            self.frozen = set(np.where(frozen_bits.astype(bool))[0])
            self.info_mask = frozen_bits == 0
        self.list_size = list_size
        self.crc_length = crc_length
        precompute_sc_indices(N)

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        paths = [_Path(N, n)]
        paths[0].L[:, 0] = llr_ch

        for i in range(N):
            l = bit_reversal_int(i, n)
            new_paths = []

            for path in paths:
                _update_llrs_path(path.L, path.B, l, n)
                llr = path.L[l, n]

                if l in self.frozen:
                    child = path.copy()
                    child.pm += self._pm_penalty(llr, 0)
                    child.u_hat[l] = 0
                    child.B[l, n] = 0
                    _update_bits_path(child.B, l, n)
                    new_paths.append(child)
                else:
                    for bit in (0, 1):
                        child = path.copy()
                        child.pm += self._pm_penalty(llr, bit)
                        child.u_hat[l] = bit
                        child.B[l, n] = bit
                        _update_bits_path(child.B, l, n)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        crc_valid = []
        for path in paths:
            if self.crc_length > 0:
                info_bits = path.u_hat[self.info_mask]
                if crc_check(info_bits, self.crc_length):
                    crc_valid.append(path)

        best = min(crc_valid, key=lambda p: p.pm) if crc_valid else paths[0]
        return best.u_hat.copy(), best.pm
