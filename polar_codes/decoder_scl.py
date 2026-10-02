"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    _SCDState,
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _lower_llr,
    _upper_llr,
)
from encoder import bit_reversal_permutation


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def _crc_update(reg, bit, poly, crc_length):
    mask = (1 << crc_length) - 1
    reg ^= int(bit) << (crc_length - 1)
    if reg & (1 << (crc_length - 1)):
        reg = ((reg << 1) ^ poly) & mask
    else:
        reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后"""
    poly = _crc_poly(crc_length)
    bits = np.asarray(info_bits, dtype=np.int8).ravel()
    reg = 0
    for b in bits:
        reg = _crc_update(reg, b, poly, crc_length)
    crc_bits = [(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)]
    return np.concatenate([bits, np.array(crc_bits, dtype=np.int8)])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    poly = _crc_poly(crc_length)
    data = np.asarray(bits, dtype=np.int8).ravel()
    reg = 0
    for b in data:
        reg = _crc_update(reg, b, poly, crc_length)
    return reg == 0


class _Path:
    __slots__ = ("pm", "B", "parent_L_id", "u_hat")

    def __init__(self, N, n):
        self.pm = 0.0
        self.B = np.full((N, n + 1), np.nan)
        self.parent_L_id = 0
        self.u_hat = np.zeros(N, dtype=np.int8)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：共享 LLR 数组）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.list_size = list_size
        self.crc_length = crc_length
        frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen = set(np.where(frozen_bits.astype(bool))[0])
        self.rev = bit_reversal_permutation(N)
        self._L_pool = [np.full((N, self.n + 1), np.nan, dtype=np.float64)]

    def _new_L(self, template):
        L = template.copy()
        self._L_pool.append(L)
        return len(self._L_pool) - 1

    def _update_llrs_path(self, L, l, B):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = int(B[j - branch_size, s + 1])
                    L[j, s + 1] = _lower_llr(L[j, s], L[j - branch_size, s], top_bit)

    def _update_bits_path(self, B, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        base_L = self._L_pool[0]
        base_L[:, 0] = llr_ch[self.rev]

        paths = [_Path(self.N, self.n)]
        paths[0].parent_L_id = 0

        decode_order = [_bit_reversed(i, self.n) for i in range(self.N)]

        for l in decode_order:
            candidates = []
            for p_idx, path in enumerate(paths):
                L = self._L_pool[path.parent_L_id].copy()
                self._update_llrs_path(L, l, path.B)
                llr_bit = L[l, self.n]
                if l in self.frozen:
                    penalty = 0.0 if llr_bit >= 0 else abs(llr_bit)
                    new_path = _Path(self.N, self.n)
                    new_path.pm = path.pm + penalty
                    new_path.B = path.B.copy()
                    new_path.B[l, self.n] = 0
                    new_path.parent_L_id = self._new_L(L)
                    new_path.u_hat = path.u_hat.copy()
                    new_path.u_hat[l] = 0
                    self._update_bits_path(new_path.B, l)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        penalty = 0.0 if (bit == 0 and llr_bit >= 0) or (bit == 1 and llr_bit < 0) else abs(llr_bit)
                        new_path = _Path(self.N, self.n)
                        new_path.pm = path.pm + penalty
                        new_path.B = path.B.copy()
                        new_path.B[l, self.n] = bit
                        new_path.parent_L_id = self._new_L(L)
                        new_path.u_hat = path.u_hat.copy()
                        new_path.u_hat[l] = bit
                        self._update_bits_path(new_path.B, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        info_idx = sorted([i for i in range(self.N) if i not in self.frozen])
        best = paths[0]
        if self.crc_length > 0:
            passed = [p for p in paths if crc_check(p.u_hat[info_idx], self.crc_length)]
            if passed:
                best = min(passed, key=lambda p: p.pm)

        return best.u_hat.astype(int), float(best.pm)
