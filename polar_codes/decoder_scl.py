"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    f_operation,
    g_operation,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_poly(crc_length)
    reg = 0
    mask = (1 << crc_length) - 1
    msb = 1 << (crc_length - 1)
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & msb:
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


def _path_metric_penalty(llr, bit):
    """与 SC 路径度量一致：不一致时加 |LLR|。"""
    hard = 0 if llr >= 0 else 1
    return 0.0 if hard == bit else abs(llr)


class _Path:
    __slots__ = ("llr_mat", "bits_mat", "pm", "parent", "branch_bit")

    def __init__(self, llr_mat, bits_mat, pm=0.0, parent=None, branch_bit=None):
        self.llr_mat = llr_mat
        self.bits_mat = bits_mat
        self.pm = pm
        self.parent = parent
        self.branch_bit = branch_bit


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [_bit_reversed(i, self.n) for i in range(N)]

    def _new_path_arrays(self, llr_ch):
        llr_mat = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        llr_mat[:, 0] = llr_ch
        bits_mat = np.zeros((self.N, self.n + 1), dtype=np.int_)
        return llr_mat, bits_mat

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    top = path.llr_mat[j, s]
                    btm = path.llr_mat[j + branch_size, s]
                    path.llr_mat[j, s + 1] = f_operation(top, btm)
                else:
                    btm = path.llr_mat[j, s]
                    top = path.llr_mat[j - branch_size, s]
                    top_bit = path.bits_mat[j - branch_size, s + 1]
                    path.llr_mat[j, s + 1] = g_operation(top, btm, top_bit)

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.bits_mat[j - branch_size, s - 1] = (
                        path.bits_mat[j, s] ^ path.bits_mat[j - branch_size, s]
                    )
                    path.bits_mat[j, s - 1] = path.bits_mat[j, s]

    def _lazy_copy(self, path):
        return _Path(path.llr_mat, path.bits_mat, path.pm, path.parent, path.branch_bit)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr0, bits0 = self._new_path_arrays(llr_ch)
        paths = [_Path(llr0, bits0, 0.0)]

        for l in self.decode_order:
            new_paths = []
            for path in paths:
                self._update_llrs(path, l)
                llr_leaf = path.llr_mat[l, self.n]

                if self.frozen_bits[l]:
                    bit = 0
                    pm = path.pm + _path_metric_penalty(llr_leaf, bit)
                    child = self._lazy_copy(path)
                    child.pm = pm
                    child.bits_mat[l, self.n] = bit
                    self._update_bits(child, l)
                    new_paths.append(child)
                else:
                    for bit in (0, 1):
                        pm = path.pm + _path_metric_penalty(llr_leaf, bit)
                        child = _Path(
                            path.llr_mat.copy(),
                            path.bits_mat.copy(),
                            pm,
                            path,
                            bit,
                        )
                        child.bits_mat[l, self.n] = bit
                        self._update_bits(child, l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        candidates = []
        for path in paths:
            u_hat = path.bits_mat[:, self.n].astype(int)
            candidates.append((path.pm, u_hat))

        if self.crc_length > 0:
            valid = [(pm, u) for pm, u in candidates if crc_check(u, self.crc_length)]
            if valid:
                pm, u_hat = min(valid, key=lambda x: x[0])
                return u_hat, pm

        pm, u_hat = min(candidates, key=lambda x: x[0])
        return u_hat, pm
