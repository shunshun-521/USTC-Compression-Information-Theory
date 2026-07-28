"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    f_operation,
    g_operation,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    _prepare_llr_channel,
    _frozen_index_set,
)
from encoder import bit_reversal_permutation


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for b in bits:
        reg ^= (int(b) << (crc_length - 1))
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) & ((1 << crc_length) - 1)) ^ (poly & ((1 << crc_length) - 1))
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_remainder(bits, poly, crc_length)
    return remainder == 0


# ==================== SCL 译码器 ====================


class _Path:
    """单条译码路径，Lazy Copy 通过 parent 引用共享 LLR/比特数组。"""

    def __init__(self, N, n, llr_ch, parent=None):
        self.N = N
        self.n = n
        self.pm = 0.0
        self.active = True
        if parent is None:
            self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
            self.C = np.zeros((N, n + 1), dtype=np.int8)
            self.L[:, 0] = llr_ch
            self.u_hat = np.zeros(N, dtype=int)
        else:
            self.L = parent.L
            self.C = parent.C
            self.u_hat = parent.u_hat.copy()


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = _frozen_index_set(self.frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length

    def _clone_path(self, path):
        new_path = _Path(self.N, self.n, None, parent=path)
        new_path.pm = path.pm
        new_path.L = path.L.copy()
        new_path.C = path.C.copy()
        return new_path

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s],
                        path.L[j, s],
                        path.C[j - branch_size, s + 1],
                    )

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.C[j - branch_size, s - 1] = path.C[j, s] ^ path.C[j - branch_size, s]
                    path.C[j, s - 1] = path.C[j, s]

    def _path_metric_penalty(self, llr, u_bit):
        """与 LLR 符号不一致时加 |LLR|。"""
        hard = 0 if llr >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回 u_hat, pm（最优路径度量）。
        """
        llr_ch = _prepare_llr_channel(llr_ch)
        paths = [_Path(self.N, self.n, llr_ch)]

        for phi in range(self.N):
            l = _bit_reversed(phi, self.n)
            candidates = []

            for path in paths:
                if not path.active:
                    continue
                self._update_llrs(path, l)
                llr_bit = path.L[l, self.n]

                if l in self.frozen_set:
                    pen = self._path_metric_penalty(llr_bit, 0)
                    new_path = self._clone_path(path)
                    new_path.pm += pen
                    new_path.C[l, self.n] = 0
                    new_path.u_hat[l] = 0
                    self._update_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for u_bit in (0, 1):
                        new_path = self._clone_path(path)
                        new_path.pm += self._path_metric_penalty(llr_bit, u_bit)
                        new_path.C[l, self.n] = u_bit
                        new_path.u_hat[l] = u_bit
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[:self.list_size]

        best_path = None
        best_pm = float("inf")

        if self.crc_length > 0:
            crc_pass = []
            for path in paths:
                info_positions = np.where(self.frozen_bits == 0)[0]
                info_bits = path.u_hat[info_positions]
                if crc_check(info_bits, self.crc_length):
                    crc_pass.append(path)
            pool = crc_pass if crc_pass else paths
        else:
            pool = paths

        for path in pool:
            if path.pm < best_pm:
                best_pm = path.pm
                best_path = path

        return best_path.u_hat.copy(), best_pm
