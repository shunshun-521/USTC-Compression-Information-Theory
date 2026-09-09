"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    f_operation,
    g_operation,
    precompute_sc_indices,
    _permute_channel_llr,
)


def _crc_generator(crc_length):
    """返回 CRC 生成多项式系数（高次到低次，含首项 1）。"""
    if crc_length == 8:
        return np.array([1, 0, 0, 0, 0, 0, 1, 1, 1], dtype=np.int8)
    if crc_length == 16:
        return np.array(
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
            dtype=np.int8,
        )
    raise ValueError(f"Unsupported CRC length: {crc_length}")


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后（GF(2) 多项式长除法）。
    """
    info_bits = np.asarray(info_bits, dtype=np.int8)
    gx = _crc_generator(crc_length)
    r = crc_length
    work = np.concatenate([info_bits, np.zeros(r, dtype=np.int8)])
    for i in range(len(info_bits)):
        if work[i] == 1:
            work[i : i + r + 1] = (work[i : i + r + 1] + gx) % 2
    return np.concatenate([info_bits, work[len(info_bits) :]])


def crc_check(bits, crc_length=8):
    """检验 bits 是否通过 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    gx = _crc_generator(crc_length)
    r = crc_length
    work = bits.copy()
    for i in range(len(bits) - r):
        if work[i] == 1:
            work[i : i + r + 1] = (work[i : i + r + 1] + gx) % 2
    return np.all(work[-r:] == 0)


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)


class SCLDecoder:
    """
    SCL 译码器（Lazy Copy 优化，Permuted SCD）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order, self.llr_layer_vec, self.bit_layer_vec = precompute_sc_indices(N)

    def _copy_path(self, src):
        dst = _Path(self.N, self.n)
        dst.L = src.L.copy()
        dst.B = src.B.copy()
        dst.pm = src.pm
        dst.u_hat = src.u_hat.copy()
        return dst

    def _update_llrs(self, path, idx):
        l = self.decode_order[idx]
        for s in self.llr_layer_vec[idx]:
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s],
                        path.L[j, s],
                        path.B[j - branch_size, s + 1],
                    )

    def _update_bits(self, path, idx, l):
        if not self.bit_layer_vec[idx]:
            return
        for s in self.bit_layer_vec[idx]:
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = (
                        path.B[j, s] ^ path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, pm)
        """
        llr = _permute_channel_llr(llr_ch)
        paths = [_Path(self.N, self.n)]
        paths[0].L[:, 0] = llr

        for idx, l in enumerate(self.decode_order):
            candidates = []
            for path in paths:
                self._update_llrs(path, idx)
                llr0 = path.L[l, self.n]

                if l in self.frozen_set:
                    new_path = self._copy_path(path)
                    new_path.pm += self._pm_penalty(llr0, 0)
                    new_path.u_hat[l] = 0
                    new_path.B[l, self.n] = 0
                    self._update_bits(new_path, idx, l)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = self._copy_path(path)
                        new_path.pm += self._pm_penalty(llr0, bit)
                        new_path.u_hat[l] = bit
                        new_path.B[l, self.n] = bit
                        self._update_bits(new_path, idx, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat, self.crc_length)]
            best = min(valid if valid else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.astype(int), best.pm
