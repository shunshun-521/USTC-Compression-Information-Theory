"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _lower_llr,
    _upper_llr,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_bits_to_int(bits):
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07, CRC-16: 0x8005
    """
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY

    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)

    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & top:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask

    crc_val = reg
    crc_bits = np.array([(crc_val >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)

    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & top:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg == 0


class _PathState:
    __slots__ = ('pm', 'L', 'B', 'u_hat')

    def __init__(self, N, n):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.u_hat = np.zeros(N, dtype=np.int8)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制状态）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.frozen_set = set(np.where(self.frozen_bits)[0])

    @staticmethod
    def _pm_add(pm, llr, bit):
        expected = 0 if llr >= 0 else 1
        if bit == expected:
            return pm
        return pm + abs(llr)

    def _update_llrs(self, path, phase):
        for stage in range(self.n - _active_llr_level(phase, self.n), self.n):
            block_size = 2 ** (stage + 1)
            branch_size = block_size // 2
            for j in range(phase, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, stage + 1] = _upper_llr(path.L[j, stage], path.L[j + branch_size, stage])
                else:
                    path.L[j, stage + 1] = _lower_llr(
                        path.L[j, stage],
                        path.L[j - branch_size, stage],
                        path.B[j - branch_size, stage + 1],
                    )

    def _update_bits(self, path, phase):
        if phase < self.N / 2:
            return
        for stage in range(self.n, self.n - _active_bit_level(phase, self.n), -1):
            block_size = 2 ** stage
            branch_size = block_size // 2
            for j in range(phase, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, stage - 1] = int(path.B[j, stage]) ^ int(path.B[j - branch_size, stage])
                    path.B[j, stage - 1] = path.B[j, stage]

    def _clone(self, path):
        new_path = _PathState(self.N, self.n)
        new_path.pm = path.pm
        new_path.u_hat = path.u_hat.copy()
        new_path.L[:] = path.L
        new_path.B[:] = path.B
        return new_path

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        rev = np.array([_bit_reversed(i, self.n) for i in range(self.N)], dtype=int)
        llr_internal = llr_ch[rev]

        paths = [_PathState(self.N, self.n)]
        paths[0].L[:, 0] = llr_internal

        for phase in [_bit_reversed(i, self.n) for i in range(self.N)]:
            candidates = []
            for path in paths:
                self._update_llrs(path, phase)
                llr = path.L[phase, self.n]

                if phase in self.frozen_set:
                    new_path = self._clone(path)
                    new_path.pm = self._pm_add(path.pm, llr, 0)
                    new_path.u_hat[phase] = 0
                    new_path.B[phase, self.n] = 0
                    self._update_bits(new_path, phase)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = self._clone(path)
                        new_path.pm = self._pm_add(path.pm, llr, bit)
                        new_path.u_hat[phase] = bit
                        new_path.B[phase, self.n] = bit
                        self._update_bits(new_path, phase)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        best_crc = None
        if self.crc_length > 0:
            for path in paths:
                info_bits = path.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    if best_crc is None or path.pm < best_crc.pm:
                        best_crc = path

        best_path = best_crc if best_crc is not None else min(paths, key=lambda p: p.pm)
        return best_path.u_hat.copy(), best_path.pm
