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
    _update_bits,
    _update_llrs,
)
from encoder import bit_reversal_permutation


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后"""
    poly = _crc_poly(crc_length)
    reg = 0
    info_bits = np.asarray(info_bits, dtype=np.int8)
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


class _Path:
    """单条 SCL 路径（Lazy Copy：复制时共享 LLR/比特数组引用）"""

    __slots__ = ("L", "B", "pm", "u_hat", "owner")

    def __init__(self, N, n, llr_ch, owner):
        self.owner = owner
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)

    def fork(self):
        child = _Path.__new__(_Path)
        child.owner = self.owner
        child.L = self.L.copy()
        child.B = self.B.copy()
        child.pm = self.pm
        child.u_hat = self.u_hat.copy()
        return child


class SCLDecoder:
    """SCL 译码器（Lazy Copy）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self._rev = bit_reversal_permutation(N)

    def _pm_penalty(self, llr_val, u):
        """路径度量增量：与 LLR 符号不一致时加 |LLR|"""
        u_hard = 0 if llr_val >= 0 else 1
        return 0.0 if u == u_hard else abs(llr_val)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_perm = llr_ch[self._rev]

        paths = [_Path(self.N, self.n, llr_perm, self)]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            candidates = []

            for path in paths:
                _update_llrs(path.L, path.B, l, self.n, self.N)
                llr_bit = path.L[l, self.n]

                if self.frozen_bits[l]:
                    pen = self._pm_penalty(llr_bit, 0)
                    path.pm += pen
                    path.B[l, self.n] = 0
                    path.u_hat[l] = 0
                    _update_bits(path.B, l, self.n, self.N)
                    candidates.append(path)
                else:
                    for u in (0, 1):
                        child = path.fork()
                        child.pm += self._pm_penalty(llr_bit, u)
                        child.B[l, self.n] = u
                        child.u_hat[l] = u
                        _update_bits(child.B, l, self.n, self.N)
                        candidates.append(child)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        # 选择最优路径
        best = min(paths, key=lambda p: p.pm)
        if self.crc_length > 0:
            info_idx = np.where(~self.frozen_bits)[0]
            passed = []
            for p in paths:
                bits = p.u_hat[info_idx]
                if crc_check(bits, self.crc_length):
                    passed.append(p)
            if passed:
                best = min(passed, key=lambda p: p.pm)
        return best.u_hat.astype(int), best.pm
