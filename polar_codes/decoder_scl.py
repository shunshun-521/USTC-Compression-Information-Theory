"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math

from decoder_sc import (
    f_operation,
    g_operation,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    _permute_channel_llr,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY

    if crc_length == 8:
        reg = 0
        for b in info_bits:
            reg ^= int(b) << 7
            for _ in range(8):
                if reg & 0x80:
                    reg = ((reg << 1) ^ poly) & 0xFF
                else:
                    reg = (reg << 1) & 0xFF
        crc_bits = np.array([(reg >> (7 - i)) & 1 for i in range(8)], dtype=int)
    else:
        reg = 0
        for b in info_bits:
            reg ^= int(b) << 15
            for _ in range(8):
                if reg & 0x8000:
                    reg = ((reg << 1) ^ poly) & 0xFFFF
                else:
                    reg = (reg << 1) & 0xFFFF
        crc_bits = np.array([(reg >> (15 - i)) & 1 for i in range(16)], dtype=int)

    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY

    if crc_length == 8:
        reg = 0
        for b in bits:
            reg ^= int(b) << 7
            for _ in range(8):
                if reg & 0x80:
                    reg = ((reg << 1) ^ poly) & 0xFF
                else:
                    reg = (reg << 1) & 0xFF
        return reg == 0
    else:
        reg = 0
        for b in bits:
            reg ^= int(b) << 15
            for _ in range(8):
                if reg & 0x8000:
                    reg = ((reg << 1) ^ poly) & 0xFFFF
                else:
                    reg = (reg << 1) & 0xFFFF
        return reg == 0


class _Path:
    __slots__ = ('pm', 'L', 'B', 'u_hat')

    def __init__(self, N, n):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        self.info_indices = np.where(self.frozen_bits == 0)[0]
        self.list_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n
        Lsz = self.list_size

        llr = _permute_channel_llr(llr_ch, N)

        paths = [_Path(N, n)]
        paths[0].L[:, 0] = llr
        paths[0].pm = 0.0

        for i in range(N):
            l = _bit_reversed(i, n)
            candidates = []

            for path in paths:
                for s in range(n - _active_llr_level(l, n), n):
                    block_size = 2 ** (s + 1)
                    branch_size = block_size // 2
                    for j in range(l, N, block_size):
                        if j % block_size < branch_size:
                            path.L[j, s + 1] = f_operation(
                                path.L[j, s], path.L[j + branch_size, s]
                            )
                        else:
                            top_bit = path.B[j - branch_size, s + 1]
                            path.L[j, s + 1] = g_operation(
                                path.B[j - branch_size, s], path.L[j, s], top_bit
                            )

                cur_llr = path.L[l, n]

                if l in self.frozen_set:
                    pen = abs(cur_llr) if cur_llr < 0 else 0.0
                    path.pm += pen
                    path.B[l, n] = 0
                    candidates.append(path)
                else:
                    for bit in (0, 1):
                        pen = 0.0
                        if bit == 0 and cur_llr < 0:
                            pen = abs(cur_llr)
                        elif bit == 1 and cur_llr >= 0:
                            pen = abs(cur_llr)
                        new_path = _Path(N, n)
                        new_path.L[:] = path.L
                        new_path.B[:] = path.B
                        new_path.pm = path.pm + pen
                        new_path.B[l, n] = bit
                        candidates.append(new_path)

            for path in candidates:
                if l >= N // 2:
                    for s in range(n, n - _active_bit_level(l, n), -1):
                        block_size = 2 ** s
                        branch_size = block_size // 2
                        for j in range(l, -1, -block_size):
                            if j % block_size >= branch_size:
                                path.B[j - branch_size, s - 1] = (
                                    path.B[j, s] ^ path.B[j - branch_size, s]
                                )
                                path.B[j, s - 1] = path.B[j, s]

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[:Lsz]

        best = paths[0]
        if self.crc_length > 0:
            valid = []
            for p in paths:
                u_dec = p.B[:, n].astype(int)
                info_bits = u_dec[self.info_indices]
                if len(info_bits) >= self.crc_length and crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                best = min(valid, key=lambda p: p.pm)

        return best.B[:, n].astype(int), best.pm
