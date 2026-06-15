"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed_index,
    _permuted_sc_decode,
    _preprocess_channel_llr,
    f_operation,
    g_operation,
)

# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    for b in bits:
        msb = (reg >> (crc_length - 1)) & 1
        reg = ((reg << 1) | int(b)) & mask
        if msb ^ int(b):
            reg ^= poly
    for _ in range(crc_length):
        msb = (reg >> (crc_length - 1)) & 1
        reg = (reg << 1) & mask
        if msb:
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


# ==================== SCL 译码器 ====================


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n, llr):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.L[:, 0] = llr

    def copy(self):
        new = _Path.__new__(_Path)
        new.L = self.L.copy()
        new.B = self.B.copy()
        new.pm = self.pm
        new.u_hat = self.u_hat.copy()
        return new


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制状态）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]
        self.decode_order = [_bit_reversed_index(i, self.n) for i in range(N)]

    @staticmethod
    def _llr_penalty(llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block = 1 << (s + 1)
            branch = block // 2
            for j in range(l, self.N, block):
                if j % block < branch:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch, s])
                else:
                    top_bit = path.B[j - branch, s + 1]
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch, s],
                        path.L[j, s],
                        top_bit,
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block = 1 << s
            branch = block // 2
            for j in range(l, -1, -block):
                if j % block >= branch:
                    path.B[j - branch, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - branch, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def _advance_bit(self, path, l, u_val):
        self._update_llrs(path, l)
        path.pm += self._llr_penalty(path.L[l, self.n], u_val)
        path.B[l, self.n] = u_val
        path.u_hat[l] = u_val
        self._update_bits(path, l)

    def decode(self, llr_ch):
        """主译码函数。"""
        llr = _preprocess_channel_llr(llr_ch)
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = _permuted_sc_decode(llr, self.frozen_bits)
            return u_hat, 0.0

        paths = [_Path(self.N, self.n, llr)]

        for l in self.decode_order:
            new_paths = []
            if self.frozen_bits[l]:
                for path in paths:
                    self._advance_bit(path, l, 0)
                    new_paths.append(path)
            else:
                for path in paths:
                    for u_val in (0, 1):
                        child = path.copy()
                        self._advance_bit(child, l, u_val)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
