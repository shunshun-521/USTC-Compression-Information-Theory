"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _frozen_to_indices,
    f_operation,
    g_operation,
)
from encoder import bit_reversal_permutation


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_divide(info_bits, poly, crc_length):
    reg = [0] * crc_length
    poly_bits = [(poly >> i) & 1 for i in range(crc_length, -1, -1)]
    poly_bits = poly_bits[1:]

    for bit in info_bits:
        feedback = bit ^ reg[0]
        reg = reg[1:] + [0]
        if feedback:
            for i in range(crc_length):
                reg[i] ^= poly_bits[i] * feedback
    return np.array(reg, dtype=int)


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07; CRC-16: 0x8005
    """
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")

    crc_bits = _crc_divide(info_bits, poly, crc_length)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=int)
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")

    remainder = _crc_divide(bits, poly, crc_length)
    return np.all(remainder == 0)


# ==================== SCL 译码器 ====================

class _Path:
    """单条译码路径（Lazy Copy）"""

    __slots__ = ("pm", "L", "B", "active")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int64)
        self.L[:, 0] = llr_ch
        self.active = True

    def copy(self):
        new_path = _Path.__new__(_Path)
        new_path.pm = self.pm
        new_path.L = self.L.copy()
        new_path.B = self.B.copy()
        new_path.active = True
        return new_path


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_indices = set(_frozen_to_indices(frozen_bits))
        self.info_positions = np.where(~np.asarray(frozen_bits, dtype=bool))[0]
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [_bit_reversed(i, self.n) for i in range(N)]
        self.br = bit_reversal_permutation(N)

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j, s],
                        path.L[j - branch_size, s],
                        path.B[j - branch_size, s + 1],
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = path.B[j, s] ^ path.B[j - branch_size, s]
                path.B[j, s - 1] = path.B[j, s]

    @staticmethod
    def _path_metric_penalty(llr_val, u_bit):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr_val)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)[self.br]
        paths = [_Path(self.N, self.n, llr_ch)]

        for phi in range(self.N):
            l = self.decode_order[phi]
            candidates = []

            for path in paths:
                if not path.active:
                    continue
                self._update_llrs(path, l)
                llr_val = path.L[l, self.n]

                if l in self.frozen_indices:
                    penalty = self._path_metric_penalty(llr_val, 0)
                    new_path = path.copy()
                    new_path.pm += penalty
                    new_path.B[l, self.n] = 0
                    self._update_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for u_bit in (0, 1):
                        new_path = path.copy()
                        new_path.pm += self._path_metric_penalty(llr_val, u_bit)
                        new_path.B[l, self.n] = u_bit
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p.B[self.info_positions, self.n], self.crc_length)
            ]
            best = min(valid if valid else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.B[:, self.n].copy(), best.pm
