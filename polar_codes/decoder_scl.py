"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    SCDecoder,
    bit_reversed,
    hard_decision,
    active_llr_level,
    active_bit_level,
    upper_llr,
    lower_llr,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_poly(crc_length)
    reg = 0
    width = crc_length
    top = 1 << (width - 1)
    mask = (1 << width) - 1
    for bit in info_bits:
        reg ^= int(bit) << (width - 1)
        for _ in range(8):
            if reg & top:
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    crc_bits = np.array([(reg >> (width - 1 - i)) & 1 for i in range(width)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    if crc_length == 0:
        return True
    return np.array_equal(crc_encode(bits[:-crc_length], crc_length), bits)


class _PathState:
    __slots__ = ('L', 'B', 'pm', 'u_hat')

    def __init__(self, N, n, llr_ch, pm=0.0):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch
        self.pm = pm
        self.u_hat = np.zeros(N, dtype=int)

    def copy(self):
        p = _PathState(self.L.shape[0], int(np.log2(self.L.shape[0])), self.L[:, 0], self.pm)
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.u_hat = self.u_hat.copy()
        return p


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen = set(np.where(np.asarray(frozen_bits, dtype=bool))[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.order = [bit_reversed(i, self.n) for i in range(N)]

    @staticmethod
    def _penalty(llr_val, u_val):
        hard = hard_decision(llr_val)
        return 0.0 if u_val == hard else abs(llr_val)

    def _update_llrs(self, path, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = upper_llr(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = lower_llr(
                        path.L[j, s],
                        path.L[j - branch_size, s],
                        int(path.B[j - branch_size, s + 1]),
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=float)

        if self.list_size == 1 and self.crc_length == 0:
            return SCDecoder(self.N, [i in self.frozen for i in range(self.N)]).decode(llr_ch), 0.0

        paths = [_PathState(self.N, self.n, llr_ch, 0.0)]

        for l in self.order:
            new_paths = []
            for path in paths:
                self._update_llrs(path, l)
                llr_val = path.L[l, self.n]

                if l in self.frozen:
                    child = path.copy()
                    child.pm += self._penalty(llr_val, 0)
                    child.B[l, self.n] = 0
                    child.u_hat[l] = 0
                    self._update_bits(child, l)
                    new_paths.append(child)
                else:
                    for u_val in (0, 1):
                        child = path.copy()
                        child.pm += self._penalty(llr_val, u_val)
                        child.B[l, self.n] = u_val
                        child.u_hat[l] = u_val
                        self._update_bits(child, l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        crc_ok = []
        for path in paths:
            if self.crc_length > 0:
                info_mask = np.array([i not in self.frozen for i in range(self.N)])
                info_bits = path.u_hat[info_mask]
                if crc_check(info_bits, self.crc_length):
                    crc_ok.append(path)
        if self.crc_length > 0 and crc_ok:
            best = min(crc_ok, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
