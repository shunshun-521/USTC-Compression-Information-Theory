"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _get_sc_precompute,
    _lower_llr_exact,
    _to_frozen_set,
    _upper_llr_exact,
)
from encoder import bit_reversed


# ==================== CRC 工具 ====================

_CRC_POLYS = {
    8: 0x07,
    16: 0x8005,
}


def _crc_divide(bits, poly, crc_length):
    reg = [0] * crc_length
    for bit in bits:
        feedback = bit ^ reg[0]
        reg = reg[1:] + [0]
        if feedback:
            for i in range(crc_length):
                if (poly >> (crc_length - 1 - i)) & 1:
                    reg[i] ^= feedback
    return np.array(reg, dtype=np.int8)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    poly = _CRC_POLYS[crc_length]
    remainder = _crc_divide(np.asarray(info_bits, dtype=np.int8), poly, crc_length)
    return np.concatenate([info_bits, remainder])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    poly = _CRC_POLYS[crc_length]
    remainder = _crc_divide(np.asarray(bits, dtype=np.int8), poly, crc_length)
    return not np.any(remainder)


# ==================== SCL 译码器 ====================

class _Path:
    __slots__ = ("pm", "L", "C")

    def __init__(self, N, n):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.C = np.full((N, n + 1), np.nan, dtype=np.float64)


class SCLDecoder:
    """
    SCL 译码器（Lazy Copy：路径分裂时复制 L/C 数组）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_set = _to_frozen_set(frozen_bits)
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order, _, _ = _get_sc_precompute(N)

    def _copy_path(self, src):
        dst = _Path(self.N, self.n)
        dst.pm = src.pm
        dst.L = src.L.copy()
        dst.C = src.C.copy()
        return dst

    def _compute_llr(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = _upper_llr_exact(
                        path.L[j, s], path.L[j + branch_size, s]
                    )
                else:
                    top_bit = 0 if np.isnan(path.C[j - branch_size, s + 1]) else int(
                        path.C[j - branch_size, s + 1]
                    )
                    path.L[j, s + 1] = _lower_llr_exact(
                        path.L[j, s], path.L[j - branch_size, s], top_bit
                    )
        return path.L[l, self.n]

    def _update_bits(self, path, l, bit):
        path.C[l, self.n] = bit
        if l >= self.N / 2:
            for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        path.C[j - branch_size, s - 1] = int(path.C[j, s]) ^ int(
                            path.C[j - branch_size, s]
                        )
                        path.C[j, s - 1] = path.C[j, s]

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        """主译码函数。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        init = _Path(self.N, self.n)
        init.L[:, 0] = llr_ch
        paths = [init]

        for phi, l in enumerate(self.decode_order):
            candidates = []
            for path in paths:
                llr = self._compute_llr(path, l)

                if l in self.frozen_set:
                    new_path = self._copy_path(path)
                    new_path.pm += self._pm_penalty(llr, 0)
                    self._update_bits(new_path, l, 0)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = self._copy_path(path)
                        new_path.pm += self._pm_penalty(llr, bit)
                        self._update_bits(new_path, l, bit)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            crc_paths = []
            for path in paths:
                info_bits = path.C[:, self.n].astype(int)[~self.frozen_bits]
                if crc_check(info_bits, self.crc_length):
                    crc_paths.append(path)
            if crc_paths:
                paths = crc_paths

        best = min(paths, key=lambda p: p.pm)
        return best.C[:, self.n].astype(int), best.pm
