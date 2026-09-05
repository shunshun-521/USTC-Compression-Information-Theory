"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    active_bit_level,
    active_llr_level,
    bit_reversed_index,
    lower_llr,
    map_channel_llr,
    upper_llr,
)


CRC_POLYS = {8: 0x07, 16: 0x8005}


def _crc_remainder(bits, crc_length):
    poly = CRC_POLYS[crc_length]
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    remainder = _crc_remainder(info_bits, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=np.int8)
    if len(bits) < crc_length:
        return False
    return _crc_remainder(bits, crc_length) == 0


class PathState:
    """单条 SCL 路径"""

    __slots__ = ("L", "B", "pm")

    def __init__(self, N, n, llr_internal):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_internal.copy()
        self.pm = 0.0

    def copy(self):
        p = PathState.__new__(PathState)
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.pm = self.pm
        return p


def _update_llrs(path, l, n):
    for s in range(n - active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, path.L.shape[0], block_size):
            if j % block_size < branch_size:
                path.L[j, s + 1] = upper_llr(path.L[j, s], path.L[j + branch_size, s])
            else:
                path.L[j, s + 1] = lower_llr(
                    path.L[j, s],
                    path.L[j - branch_size, s],
                    int(path.B[j - branch_size, s + 1]),
                )


def _update_bits(path, l, n):
    if l < path.L.shape[0] // 2:
        return
    for s in range(n, n - active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                path.B[j, s - 1] = path.B[j, s]


def _path_penalty(llr, u):
    hard = 0 if llr >= 0 else 1
    return 0.0 if u == hard else abs(llr)


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

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
        llr_internal = map_channel_llr(llr_ch)
        paths = [PathState(self.N, self.n, llr_internal)]

        for l in [bit_reversed_index(i, self.n) for i in range(self.N)]:
            candidates = []
            for path in paths:
                _update_llrs(path, l, self.n)
                llr = path.L[l, self.n]

                if l in self.frozen_set:
                    new_path = path.copy()
                    new_path.pm += _path_penalty(llr, 0)
                    new_path.B[l, self.n] = 0
                    _update_bits(new_path, l, self.n)
                    candidates.append(new_path)
                else:
                    for u in (0, 1):
                        new_path = path.copy()
                        new_path.pm += _path_penalty(llr, u)
                        new_path.B[l, self.n] = u
                        _update_bits(new_path, l, self.n)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        best_crc = None
        best_any = min(paths, key=lambda p: p.pm)

        if self.crc_length > 0:
            for path in paths:
                u = path.B[:, self.n].astype(int)
                info_bits = u[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    if best_crc is None or path.pm < best_crc.pm:
                        best_crc = path

        chosen = best_crc if best_crc is not None else best_any
        u_hat = chosen.B[:, self.n].astype(int)
        return u_hat, chosen.pm
