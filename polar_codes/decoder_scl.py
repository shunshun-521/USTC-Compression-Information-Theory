"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    prepare_decoder_llrs,
    _bit_reversed_index,
    _active_llr_level,
    _active_bit_level,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg = (reg << 1) | int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    return reg & ((1 << crc_length) - 1)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY if crc_length == 16 else None
    if poly is None:
        raise ValueError("crc_length must be 8 or 16")

    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 的 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY if crc_length == 16 else None
    if poly is None:
        raise ValueError("crc_length must be 8 or 16")
    return _crc_remainder(bits, poly, crc_length) == 0


class _PathState:
    __slots__ = ("pm", "L", "B")

    def __init__(self, N, n):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length

    def _path_llr(self, path, phase):
        for s in range(self.n - _active_llr_level(phase, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(phase, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s], path.L[j, s], path.B[j - branch_size, s + 1]
                    )
        return path.L[phase, self.n]

    def _path_propagate_bits(self, path, phase, bit):
        path.B[phase, self.n] = bit
        if phase >= self.N / 2:
            for s in range(self.n, self.n - _active_bit_level(phase, self.n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(phase, -1, -block_size):
                    if j % block_size >= branch_size:
                        path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                        path.B[j, s - 1] = path.B[j, s]

    def _path_bits(self, path):
        return path.B[:, self.n].astype(int)

    def decode(self, llr_ch):
        llr = prepare_decoder_llrs(llr_ch)
        paths = [_PathState(self.N, self.n) for _ in range(self.list_size)]
        active = [paths[0]]
        active[0].L[:, 0] = llr

        for phase in [_bit_reversed_index(i, self.n) for i in range(self.N)]:
            candidates = []

            for path in active:
                llr_bit = self._path_llr(path, phase)

                if phase in self.frozen_set:
                    penalty = abs(llr_bit) if llr_bit < 0 else 0.0
                    candidates.append((path.pm + penalty, path, 0))
                else:
                    pm0 = path.pm + (0.0 if llr_bit >= 0 else abs(llr_bit))
                    pm1 = path.pm + (abs(llr_bit) if llr_bit >= 0 else 0.0)
                    candidates.append((pm0, path, 0))
                    candidates.append((pm1, path, 1))

            candidates.sort(key=lambda x: x[0])
            selected = candidates[: self.list_size]

            new_active = []
            for pm, parent, bit in selected:
                child = _PathState(self.N, self.n)
                child.pm = pm
                child.L = parent.L.copy()
                child.B = parent.B.copy()
                self._path_propagate_bits(child, phase, bit)
                new_active.append(child)

            active = new_active

        if self.crc_length > 0:
            crc_paths = [p for p in active if crc_check(self._path_bits(p), self.crc_length)]
            if crc_paths:
                active = crc_paths

        best = min(active, key=lambda p: p.pm)
        u_hat = self._path_bits(best)
        u_hat[self.frozen_bits] = 0
        return u_hat, best.pm
