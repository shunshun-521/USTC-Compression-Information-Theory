"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _lower_llr,
    _update_bits,
    _upper_llr,
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
    reg = 0
    mask = (1 << crc_length) - 1
    msb = 1 << (crc_length - 1)
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & msb:
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    crc_bits = np.array(
        [(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    if len(bits) < crc_length:
        return False
    return np.array_equal(crc_encode(bits[:-crc_length], crc_length), bits)


def _update_llrs_path(L, B, l, n, N):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
            else:
                top_llr = L[j - branch_size, s]
                btm_llr = L[j, s]
                top_bit = B[j - branch_size, s + 1]
                L[j, s + 1] = _lower_llr(btm_llr, top_llr, top_bit)


class _Path:
    __slots__ = ("L", "B", "pm")

    def __init__(self, n, N, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = llr_ch
        self.pm = 0.0


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    @staticmethod
    def _pm_update(pm, llr, u):
        hard = 0 if llr >= 0 else 1
        if u != hard:
            pm += abs(llr)
        return pm

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        paths = [_Path(n, N, llr_ch)]

        for i in range(N):
            l = _bit_reversed(i, n)
            new_paths = []
            for path in paths:
                _update_llrs_path(path.L, path.B, l, n, N)
                llr = path.L[l, n]
                if self.frozen_bits[l]:
                    pm = self._pm_update(path.pm, llr, 0)
                    path.pm = pm
                    path.B[l, n] = 0
                    _update_bits(path.B, l, n, N)
                    new_paths.append(path)
                else:
                    for u in (0, 1):
                        child = _Path(n, N, llr_ch)
                        child.L[:] = path.L
                        child.B[:] = path.B
                        child.pm = self._pm_update(path.pm, llr, u)
                        child.B[l, n] = u
                        _update_bits(child.B, l, n, N)
                        new_paths.append(child)
            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        paths.sort(key=lambda p: p.pm)
        best = paths[0]
        u_hat = best.B[:, n].astype(int)

        if self.crc_length > 0:
            for p in paths:
                ib = p.B[:, n][self.info_indices]
                if crc_check(ib, self.crc_length):
                    return p.B[:, n].astype(int), p.pm

        return u_hat, best.pm
