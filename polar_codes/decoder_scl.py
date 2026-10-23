"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    active_bit_level,
    active_llr_level,
    bit_reversed,
    f_operation,
    g_operation,
)


def _crc_process(bits, crc_length, init=0):
    poly = 0x07 if crc_length == 8 else 0x8005
    reg = init
    mask = (1 << crc_length) - 1
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    rem = _crc_process(info_bits, crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 的 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    rem = _crc_process(bits, crc_length)
    return rem == 0


class _PathState:
    __slots__ = ("pm", "L", "B")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = llr_ch


def _clone_path(path):
    new = _PathState(path.L.shape[0], path.L.shape[1] - 1, path.L[:, 0])
    new.pm = path.pm
    new.L = path.L.copy()
    new.B = path.B.copy()
    return new


def _update_llrs_path(L, B, l, n):
    start = n - active_llr_level(l, n)
    for s in range(start, n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, L.shape[0], block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = g_operation(
                    L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                )


def _update_bits_path(B, l, n):
    if l < B.shape[0] // 2:
        return
    end = n - active_bit_level(l, n)
    for s in range(n, end, -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def _metric_add(llr, u):
    u_hard = 0 if llr >= 0 else 1
    return 0.0 if u == u_hard else abs(llr)


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_PathState(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            new_paths = []

            for path in paths:
                _update_llrs_path(path.L, path.B, l, self.n)
                llr = path.L[l, self.n]

                if l in self.frozen_set:
                    p = _clone_path(path)
                    p.pm += _metric_add(llr, 0)
                    p.B[l, self.n] = 0
                    _update_bits_path(p.B, l, self.n)
                    new_paths.append(p)
                else:
                    for u in (0, 1):
                        p = _clone_path(path)
                        p.pm += _metric_add(llr, u)
                        p.B[l, self.n] = u
                        _update_bits_path(p.B, l, self.n)
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        paths.sort(key=lambda p: p.pm)
        best = paths[0]
        u_hat = best.B[:, self.n].astype(int)

        if self.crc_length > 0:
            for p in paths:
                info = p.B[:, self.n].astype(int)[self.info_indices]
                if crc_check(info, self.crc_length):
                    u_hat = p.B[:, self.n].astype(int)
                    best = p
                    break

        return u_hat, best.pm
