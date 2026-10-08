"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation
from decoder_sc import (
    active_bit_level,
    active_llr_level,
    bit_reversed,
    lower_llr,
    upper_llr,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def _crc_remainder(bits, crc_length):
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    crc = 0
    for bit in bits:
        crc ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if crc & (1 << (crc_length - 1)):
                crc = ((crc << 1) ^ poly) & mask
            else:
                crc = (crc << 1) & mask
    return crc


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    crc = _crc_remainder(info_bits, crc_length)
    crc_bits = [(crc >> (crc_length - 1 - i)) & 1 for i in range(crc_length)]
    return np.concatenate([np.asarray(info_bits, dtype=int), np.asarray(crc_bits, dtype=int)])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    payload = bits[:-crc_length]
    expected = crc_encode(payload, crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


class _Path:
    __slots__ = ("pm", "L", "B")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch


def _update_llrs(path, l, n, N):
    for s in range(n - active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                path.L[j, s + 1] = upper_llr(path.L[j, s], path.L[j + branch_size, s])
            else:
                path.L[j, s + 1] = lower_llr(
                    path.L[j, s], path.L[j - branch_size, s], int(path.B[j - branch_size, s + 1])
                )


def _update_bits(path, l, n, N):
    if l < N // 2:
        return
    for s in range(n, n - active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                path.B[j, s - 1] = path.B[j, s]


def _llr_penalty(llr, u):
    u_hard = 0 if llr >= 0 else 1
    return 0.0 if u == u_hard else abs(llr)


class SCLDecoder:
    """SCL 译码器（Permuted SCD + 路径度量）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_ch = llr_ch[bit_reversal_permutation(self.N)]
        N, n = self.N, self.n

        paths = [_Path(N, n, llr_ch)]

        for i in range(N):
            l = bit_reversed(i, n)
            candidates = []
            for path in paths:
                _update_llrs(path, l, n, N)
                llr = path.L[l, n]
                if self.frozen_bits[l]:
                    pen = _llr_penalty(llr, 0)
                    path.pm += pen
                    path.B[l, n] = 0
                    _update_bits(path, l, n, N)
                    candidates.append(path)
                else:
                    for u in (0, 1):
                        new = _Path(N, n, llr_ch)
                        new.L = path.L.copy()
                        new.B = path.B.copy()
                        new.pm = path.pm + _llr_penalty(llr, u)
                        new.B[l, n] = u
                        _update_bits(new, l, n, N)
                        candidates.append(new)
            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        crc_pass = []
        for p in paths:
            u_hat = p.B[:, n].astype(int)
            if crc_check(u_hat[self.info_indices], self.crc_length):
                crc_pass.append(p)
        best = min(crc_pass or paths, key=lambda p: p.pm)
        return best.B[:, n].astype(int), best.pm
