"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import math

import numpy as np

from decoder_sc import (
    active_bit_level,
    active_llr_level,
    bit_reversed,
    f_operation,
    g_operation,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _bits_to_bytes(bits):
    """将比特序列按 MSB-first 打包为字节。"""
    bits = np.asarray(bits, dtype=int)
    nbytes = (len(bits) + 7) // 8
    out = bytearray(nbytes)
    for i, bit in enumerate(bits):
        if bit:
            out[i // 8] |= 1 << (7 - i % 8)
    return bytes(out)


def _crc8_bytes(data, poly=CRC8_POLY):
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ poly) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


def _crc16_bytes(data, poly=CRC16_POLY):
    crc = 0
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ poly) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    data = _bits_to_bytes(info_bits)
    if crc_length == 8:
        remainder = _crc8_bytes(data)
    else:
        remainder = _crc16_bytes(data)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 的 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    data = _bits_to_bytes(bits)
    if crc_length == 8:
        return _crc8_bytes(data) == 0
    return _crc16_bytes(data) == 0


class Path:
    """SCL 单条路径。"""

    __slots__ = ("pm", "u_hat", "L", "B")

    def __init__(self, N, n):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _update_llrs(self, path, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size >> 1
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s],
                        path.L[j, s],
                        path.B[j - branch_size, s + 1],
                    )

    def _update_bits(self, path, l, bit):
        path.B[l, self.n] = bit
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size >> 1
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = (
                        int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                    ) % 2
                    path.B[j, s - 1] = path.B[j, s]

    def _path_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        """主译码函数。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [Path(self.N, self.n)]
        paths[0].L[:, 0] = llr_ch

        for phi in range(self.N):
            l = bit_reversed(phi, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, self.n]

                if self.frozen_bits[l]:
                    new_path = copy.copy(path)
                    new_path.L = path.L.copy()
                    new_path.B = path.B.copy()
                    new_path.u_hat = path.u_hat.copy()
                    new_path.pm += self._path_penalty(llr, 0)
                    new_path.u_hat[l] = 0
                    self._update_bits(new_path, l, 0)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = copy.copy(path)
                        new_path.L = path.L.copy()
                        new_path.B = path.B.copy()
                        new_path.u_hat = path.u_hat.copy()
                        new_path.pm += self._path_penalty(llr, bit)
                        new_path.u_hat[l] = bit
                        self._update_bits(new_path, l, bit)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        return self._select_best_path(paths)

    def _select_best_path(self, paths):
        if self.crc_length > 0:
            passed = []
            for path in paths:
                info_bits = path.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    passed.append(path)
            if passed:
                best = min(passed, key=lambda p: p.pm)
                return best.u_hat.copy(), best.pm

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
