"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation
from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed_scalar,
    _lower_llr,
    _upper_llr,
)


def _crc_divide(data, poly, crc_len):
    """GF(2) CRC 除法，返回余数比特"""
    reg = np.zeros(crc_len, dtype=np.int8)
    for bit in data:
        feedback = reg[0] ^ bit
        reg[:-1] = reg[1:]
        reg[-1] = 0
        if feedback:
            for i in range(crc_len):
                if (poly >> (crc_len - 1 - i)) & 1:
                    reg[i] ^= feedback
    return reg


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    r=8: CRC-8 (0x07); r=16: CRC-16 (0x8005)
    """
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")

    remainder = _crc_divide(info_bits, poly, crc_length)
    return np.concatenate([info_bits, remainder])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确"""
    bits = np.asarray(bits, dtype=np.int8)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")

    remainder = _crc_divide(bits, poly, crc_length)
    return np.all(remainder == 0)


class _Path:
    """单条 SCL 路径（Lazy Copy：共享数组引用）"""
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)

    def copy(self):
        p = _Path.__new__(_Path)
        p.L = self.L
        p.B = self.B.copy()
        p.pm = self.pm
        p.u_hat = self.u_hat.copy()
        return p

    def deep_copy_arrays(self):
        self.L = self.L.copy()
        self.B = self.B.copy()


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self._br = bit_reversal_permutation(N)
        self.frozen_bits = np.asarray(frozen_bits)
        if self.frozen_bits.dtype == bool:
            self.frozen_set = set(np.where(self.frozen_bits)[0])
        else:
            self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [_bit_reversed_scalar(i, self.n) for i in range(N)]

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = _upper_llr(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = _lower_llr(
                        path.L[j, s],
                        path.L[j - branch_size, s],
                        int(path.B[j - branch_size, s + 1]),
                    )

    def _update_bits(self, path, l, bit):
        path.B[l, self.n] = bit
        if l >= self.N // 2:
            for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                            path.B[j - branch_size, s]
                        )
                        path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr, bit):
        """路径度量惩罚：与 LLR 硬判决不一致时加 |LLR|"""
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_ch = llr_ch[self._br]
        paths = [_Path(self.N, self.n, llr_ch)]

        for l in self.decode_order:
            new_paths = []
            for path in paths:
                path.deep_copy_arrays()
                self._update_llrs(path, l)
                llr = path.L[l, self.n]

                if l in self.frozen_set:
                    pen = self._pm_penalty(llr, 0)
                    path.pm += pen
                    path.u_hat[l] = 0
                    self._update_bits(path, l, 0)
                    new_paths.append(path)
                else:
                    for bit in (0, 1):
                        p = path.copy()
                        p.deep_copy_arrays()
                        p.pm += self._pm_penalty(llr, bit)
                        p.u_hat[l] = bit
                        self._update_bits(p, l, bit)
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat, self.crc_length)]
            best = min(valid if valid else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
