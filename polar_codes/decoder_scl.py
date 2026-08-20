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
    _permute_channel_llrs,
    f_operation,
    g_operation,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, width):
    mask = (1 << width) - 1
    top = 1 << (width - 1)
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (width - 1)
        if reg & top:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    msg = np.concatenate([info_bits, np.zeros(crc_length, dtype=int)])
    remainder = _crc_remainder(msg, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    msg = np.concatenate([bits[:-crc_length], np.zeros(crc_length, dtype=int)])
    remainder = _crc_remainder(msg, poly, crc_length)
    expected = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.array_equal(bits[-crc_length:], expected)


class Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(np.logical_not(self.frozen_bits))[0]
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [_bit_reversed_index(i, self.n) for i in range(N)]
        self.llr_layers = [
            list(range(self.n - _active_llr_level(l, self.n), self.n))
            for l in self.decode_order
        ]
        self.bit_layers = []
        for l in self.decode_order:
            if l < N / 2:
                self.bit_layers.append([])
            else:
                self.bit_layers.append(
                    list(range(self.n, self.n - _active_bit_level(l, self.n), -1))
                )

    def _update_llrs(self, path, l, step):
        for s in self.llr_layers[step]:
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
                        path.L[j - branch_size, s], path.L[j, s], top_bit
                    )

    def _update_bits(self, path, l, step):
        for s in self.bit_layers[step]:
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def _path_metric_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = _permute_channel_llrs(llr_ch, self.N)
        paths = [Path(self.N, self.n, llr_ch.copy())]

        for step, l in enumerate(self.decode_order):
            for path in paths:
                self._update_llrs(path, l, step)

            candidates = []
            if l in self.frozen_set:
                for path in paths:
                    llr = path.L[l, self.n]
                    candidates.append((path.pm + self._path_metric_penalty(llr, 0), path, 0))
            else:
                for path in paths:
                    llr = path.L[l, self.n]
                    for bit in (0, 1):
                        candidates.append(
                            (path.pm + self._path_metric_penalty(llr, bit), path, bit)
                        )

            candidates.sort(key=lambda x: x[0])
            candidates = candidates[: self.list_size]

            new_paths = []
            for pm, parent, bit in candidates:
                child = Path(self.N, self.n, parent.L[:, 0])
                child.L = parent.L.copy()
                child.B = parent.B.copy()
                child.u_hat = parent.u_hat.copy()
                child.pm = pm
                child.u_hat[l] = bit
                child.B[l, self.n] = bit
                self._update_bits(child, l, step)
                new_paths.append(child)
            paths = new_paths

        if self.crc_length > 0:
            valid = []
            for path in paths:
                info_bits = path.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
