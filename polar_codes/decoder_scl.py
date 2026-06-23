"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    prepare_channel_llr,
    _bit_reversed_index,
    _active_llr_level,
    _active_bit_level,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    """CRC 移位寄存器余数计算。"""
    reg = np.zeros(crc_length, dtype=np.int8)
    for bit in bits:
        feedback = int(bit) ^ reg[0]
        reg[:-1] = reg[1:]
        reg[-1] = 0
        if feedback:
            for i in range(crc_length):
                if (poly >> (crc_length - 1 - i)) & 1:
                    reg[i] ^= 1
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    return np.concatenate([info_bits, remainder.astype(np.int8)])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(bits, poly, crc_length)
    return np.all(remainder == 0)


class Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _copy_path(self, src):
        dst = Path(self.N, self.n)
        dst.L[:] = src.L
        dst.B[:] = src.B
        dst.pm = src.pm
        dst.u_hat[:] = src.u_hat
        return dst

    @staticmethod
    def _pm_penalty(llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def _update_llrs(self, paths, l):
        for path in paths:
            for s in range(self.n - _active_llr_level(l, self.n), self.n):
                block_size = 1 << (s + 1)
                branch_size = block_size // 2
                for j in range(l, self.N, block_size):
                    if j % block_size < branch_size:
                        path.L[j, s + 1] = f_operation(
                            path.L[j, s], path.L[j + branch_size, s]
                        )
                    else:
                        top_bit = int(path.B[j - branch_size, s + 1])
                        path.L[j, s + 1] = g_operation(
                            path.L[j - branch_size, s],
                            path.L[j, s],
                            top_bit,
                        )

    def _update_bits(self, paths, l):
        if l < self.N // 2:
            return
        for path in paths:
            for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                            path.B[j - branch_size, s]
                        )
                        path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        llr = prepare_channel_llr(llr_ch)
        paths = [Path(self.N, self.n)]
        paths[0].L[:, 0] = llr

        for phi in range(self.N):
            l = _bit_reversed_index(phi, self.n)
            self._update_llrs(paths, l)

            if self.frozen_bits[l]:
                for path in paths:
                    path.pm += self._pm_penalty(path.L[l, self.n], 0)
                    path.u_hat[l] = 0
                    path.B[l, self.n] = 0
                candidates = paths
            else:
                candidates = []
                for path in paths:
                    llr_bit = path.L[l, self.n]
                    for bit in (0, 1):
                        new_path = self._copy_path(path)
                        new_path.pm += self._pm_penalty(llr_bit, bit)
                        new_path.u_hat[l] = bit
                        new_path.B[l, self.n] = bit
                        candidates.append(new_path)
                candidates.sort(key=lambda p: p.pm)
                candidates = candidates[: self.list_size]

            self._update_bits(candidates, l)
            paths = candidates

        if self.crc_length > 0:
            valid = []
            for path in paths:
                info_bits = path.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            best = min(valid if valid else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
