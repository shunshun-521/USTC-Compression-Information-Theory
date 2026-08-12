"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from encoder import bit_reversed
from decoder_sc import (
    _active_llr_level,
    _active_bit_level,
    _upper_llr,
    _lower_llr,
    f_operation,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class SCLPath:
    def __init__(self, N, n):
        self.N = N
        self.n = n
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.u_hat = np.zeros(N, dtype=int)
        self.pm = 0.0

    def copy_from(self, other):
        self.L[:] = other.L
        self.B[:] = other.B
        self.u_hat[:] = other.u_hat
        self.pm = other.pm


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length

        self.info_indices = np.where(~self.frozen_bits)[0]
        if crc_length > 0:
            self.crc_info_indices = self.info_indices[:len(self.info_indices) - crc_length]
        else:
            self.crc_info_indices = self.info_indices

    def _update_llrs(self, path, l):
        n = self.n
        N = self.N
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = _upper_llr(path.L[j, s], path.L[j + branch_size, s])
                else:
                    top_bit = path.B[j - branch_size, s + 1]
                    if np.isnan(top_bit):
                        top_bit = 0
                    path.L[j, s + 1] = _lower_llr(
                        path.L[j, s],
                        path.L[j - branch_size, s],
                        int(top_bit),
                    )

    def _update_bits(self, path, l):
        n = self.n
        N = self.N
        if l < N // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr_val, u_val):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_val == hard else abs(llr_val)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n

        paths = [SCLPath(N, n)]
        paths[0].L[:, 0] = llr_ch

        for phi_nat in range(N):
            l = bit_reversed(phi_nat, n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr_val = path.L[l, n]
                if np.isnan(llr_val):
                    llr_val = 0.0

                if self.frozen_bits[l]:
                    new_path = SCLPath(N, n)
                    new_path.copy_from(path)
                    penalty = self._pm_penalty(llr_val, 0)
                    new_path.pm += penalty
                    new_path.u_hat[l] = 0
                    new_path.B[l, n] = 0
                    self._update_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for u_val in (0, 1):
                        new_path = SCLPath(N, n)
                        new_path.copy_from(path)
                        penalty = self._pm_penalty(llr_val, u_val)
                        new_path.pm += penalty
                        new_path.u_hat[l] = u_val
                        new_path.B[l, n] = u_val
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[:self.list_size]

        if self.crc_length > 0:
            crc_pass = [
                p for p in paths
                if crc_check(p.u_hat[self.crc_info_indices], self.crc_length)
            ]
            best = min(crc_pass if crc_pass else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
