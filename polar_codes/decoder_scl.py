"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math

from encoder import bit_reversal_permutation
from decoder_sc import (
    f_operation,
    g_operation,
    _boxplus_f,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    _frozen_indices,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    """计算 CRC 余数（不含附加校验位）。"""
    reg = 0
    top_bit = 1 << (crc_length - 1)
    mask = (1 << crc_length) - 1
    for bit in bits:
        feedback = ((reg >> (crc_length - 1)) ^ int(bit)) & 1
        reg = (reg << 1) & mask
        if feedback:
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)

    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """
    检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。
    """
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(bits[:-crc_length], poly, crc_length)
    expected = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.array_equal(bits[-crc_length:], expected)


class _Path:
    """单条 SCL 译码路径。"""

    def __init__(self, N, n):
        self.N = N
        self.n = n
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.active = True


class SCLDecoder:
    """
    SCL 译码器（Lazy Copy：路径共享 LLR/比特矩阵引用）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = _frozen_indices(self.frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]
        self.rev = bit_reversal_permutation(N)

    def _init_path(self, llr_ch):
        path = _Path(self.N, self.n)
        path.L[:, 0] = llr_ch[self.rev]
        return path

    def _update_llrs_for_bit(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = _boxplus_f(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s],
                        path.L[j, s],
                        path.B[j - branch_size, s + 1],
                    )

    def _update_bits_for_bit(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def _path_metric_penalty(self, llr, u_bit):
        """路径度量惩罚：判决与 LLR 符号不一致时加 |LLR|。"""
        hard = 0 if llr >= 0 else 1
        if u_bit == hard:
            return 0.0
        return abs(llr)

    def _crc_pass(self, path):
        if self.crc_length == 0:
            return True
        info_bits = path.u_hat[self.info_indices]
        if len(info_bits) < self.crc_length:
            return False
        return crc_check(info_bits, self.crc_length)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, pm)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [self._init_path(llr_ch)]

        decode_order = [_bit_reversed(i, self.n) for i in range(self.N)]

        for phi_nat in range(self.N):
            l = decode_order[phi_nat]
            candidates = []

            for path in paths:
                self._update_llrs_for_bit(path, l)
                llr_bit = path.L[l, self.n]

                if l in self.frozen_set:
                    penalty = self._path_metric_penalty(llr_bit, 0)
                    new_path = self._clone_path(path)
                    new_path.pm += penalty
                    new_path.B[l, self.n] = 0
                    new_path.u_hat[l] = 0
                    self._update_bits_for_bit(new_path, l)
                    candidates.append(new_path)
                else:
                    for u_bit in (0, 1):
                        new_path = self._clone_path(path)
                        penalty = self._path_metric_penalty(llr_bit, u_bit)
                        new_path.pm += penalty
                        new_path.B[l, self.n] = u_bit
                        new_path.u_hat[l] = u_bit
                        self._update_bits_for_bit(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[:self.list_size]

        crc_paths = [p for p in paths if self._crc_pass(p)]
        if crc_paths:
            best = min(crc_paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm

    def _clone_path(self, path):
        """Lazy copy：复制路径状态（浅拷贝矩阵，已修改单元独立）。"""
        new_path = _Path(self.N, self.n)
        new_path.L = path.L.copy()
        new_path.B = path.B.copy()
        new_path.pm = path.pm
        new_path.u_hat = path.u_hat.copy()
        return new_path
