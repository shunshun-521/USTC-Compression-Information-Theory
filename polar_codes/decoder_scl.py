"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math
from decoder_sc import (
    bit_reversed,
    g_operation,
    prepare_llr_for_decoder,
    _active_llr_level,
    _active_bit_level,
    _logdomain_sum,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    """计算 CRC 余数（不含发送端补零）"""
    reg = 0
    for b in bits:
        reg ^= (int(b) << (crc_length - 1))
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8 (0x07) 或 CRC-16 (0x8005)
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 尾部 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(bits, poly, crc_length)
    return remainder == 0


class _Path:
    def __init__(self, N, n):
        self.N = N
        self.n = n
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.u_hat = np.zeros(N, dtype=int)
        self.pm = 0.0

    def copy(self):
        p = _Path(self.N, self.n)
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.u_hat = self.u_hat.copy()
        p.pm = self.pm
        return p


class SCLDecoder:
    """SCL 译码器（列表路径复制实现）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits > 0)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def _update_llrs_for_phase(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    a, b = path.L[j, s], path.L[j + branch_size, s]
                    path.L[j, s + 1] = _logdomain_sum(a + b, 0.0) - _logdomain_sum(a, b)
                else:
                    u_bit = path.B[j - branch_size, s + 1]
                    a, b = path.L[j, s], path.L[j - branch_size, s]
                    path.L[j, s + 1] = a + b if u_bit == 0 else a - b

    def _update_bits_for_phase(self, path, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr, u_bit):
        u_nat = 0 if llr >= 0 else 1
        return 0.0 if u_bit == u_nat else abs(llr)

    def decode(self, llr_ch):
        llr = prepare_llr_for_decoder(llr_ch, self.N)

        paths = [_Path(self.N, self.n)]
        paths[0].L[:, 0] = llr

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            new_paths = []

            for path in paths:
                self._update_llrs_for_phase(path, l)
                llr_leaf = path.L[l, self.n]

                if l in self.frozen_set:
                    u_bit = 0
                    path.pm += self._pm_penalty(llr_leaf, u_bit)
                    path.u_hat[l] = u_bit
                    path.B[l, self.n] = u_bit
                    self._update_bits_for_phase(path, l)
                    new_paths.append(path)
                else:
                    for u_bit in (0, 1):
                        p = path.copy()
                        p.pm += self._pm_penalty(llr_leaf, u_bit)
                        p.u_hat[l] = u_bit
                        p.B[l, self.n] = u_bit
                        self._update_bits_for_phase(p, l)
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[:self.list_size]

        if self.crc_length > 0:
            crc_pass = []
            for p in paths:
                info_bits = p.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_pass.append(p)
            if crc_pass:
                paths = crc_pass

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
