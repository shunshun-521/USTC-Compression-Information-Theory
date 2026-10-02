"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import math
import numpy as np
from decoder_sc import _channel_llr_to_decoder
from decoder_utils_internal import (
    active_bit_level,
    active_llr_level,
    bit_reversed,
    hard_decision,
    lower_llr,
    upper_llr,
)

# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, degree):
    reg = 0
    for b in bits:
        reg ^= int(b) << (degree - 1)
        for _ in range(8 if degree == 8 else 16):
            if reg & (1 << (degree - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << degree) - 1)
            else:
                reg = (reg << 1) & ((1 << degree) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    if crc_length == 8:
        rem = _crc_remainder(info_bits, _CRC8_POLY, 8)
        crc_bits = np.array([(rem >> (7 - i)) & 1 for i in range(8)], dtype=int)
    elif crc_length == 16:
        rem = _crc_remainder(info_bits, _CRC16_POLY, 16)
        crc_bits = np.array([(rem >> (15 - i)) & 1 for i in range(16)], dtype=int)
    else:
        raise ValueError("crc_length must be 8 or 16")
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=int).ravel()
    if crc_length == 0:
        return True
    payload = bits[:-crc_length]
    expected = crc_encode(payload, crc_length)
    return np.array_equal(bits, expected)


# ==================== SCL 译码器 ====================


class _PathState:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（Permuted SCD + 路径度量）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.list_size = list_size
        self.crc_length = crc_length
        frozen_bits = np.asarray(frozen_bits)
        self.frozen_set = set(np.where(frozen_bits.astype(bool))[0])
        self.info_indices = np.where(~frozen_bits.astype(bool))[0]

    def _update_llrs(self, path, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
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
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr, u_bit):
        u_from_llr = 0 if llr >= 0 else 1
        return 0.0 if u_bit == u_from_llr else abs(llr)

    def decode(self, llr_ch):
        llr_ch = _channel_llr_to_decoder(llr_ch)
        paths = [_PathState(self.N, self.n)]
        paths[0].L[:, 0] = llr_ch

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            new_paths = []
            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, self.n]
                if l in self.frozen_set:
                    pen = self._pm_penalty(llr, 0)
                    child = copy.deepcopy(path)
                    child.pm += pen
                    child.B[l, self.n] = 0
                    child.u_hat[l] = 0
                    self._update_bits(child, l)
                    new_paths.append(child)
                else:
                    for u_bit in (0, 1):
                        child = copy.deepcopy(path)
                        child.pm += self._pm_penalty(llr, u_bit)
                        child.B[l, self.n] = u_bit
                        child.u_hat[l] = u_bit
                        self._update_bits(child, l)
                        new_paths.append(child)
            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        # 选择最优路径
        valid = []
        for p in paths:
            if self.crc_length > 0:
                info_bits = p.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            else:
                valid.append(p)
        if valid:
            best = min(valid, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
