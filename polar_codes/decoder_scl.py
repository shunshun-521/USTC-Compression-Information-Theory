"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed_index,
    _lower_llr_exact,
    _prepare_llr,
    _to_frozen_set,
    _upper_llr_exact,
    f_operation,
    g_operation,
)


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    使用 CRC-8 (0x07) 或 CRC-16 (0x8005)。
    """
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    for bit in info_bits:
        fb = ((reg >> (crc_length - 1)) ^ int(bit)) & 1
        reg = (reg << 1) & ((1 << crc_length) - 1)
        if fb:
            reg ^= poly

    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 是否满足 CRC 约束。"""
    bits = np.asarray(bits, dtype=int)
    return np.array_equal(bits, crc_encode(bits[:-crc_length], crc_length))


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（PSCD + Lazy Copy）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_set = _to_frozen_set(frozen_bits)
        self.frozen_bits = np.asarray(frozen_bits).astype(bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.decode_order = [_bit_reversed_index(i, self.n) for i in range(N)]

    def _clone(self, path):
        child = _Path(self.N, self.n)
        child.L = path.L.copy()
        child.B = path.B.copy()
        child.pm = path.pm
        child.u_hat = path.u_hat.copy()
        return child

    @staticmethod
    def _penalty(llr, u_bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr)

    def _update_llrs(self, path, l_idx):
        n = self.n
        N = self.N
        for s in range(n - _active_llr_level(l_idx, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size >> 1
            for j in range(l_idx, N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = _upper_llr_exact(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = _lower_llr_exact(
                        path.L[j, s],
                        path.L[j - branch_size, s],
                        int(path.B[j - branch_size, s + 1]),
                    )

    def _update_bits(self, path, l_idx, u_bit):
        path.B[l_idx, self.n] = u_bit
        if l_idx >= self.N // 2:
            for s in range(self.n, self.n - _active_bit_level(l_idx, self.n), -1):
                block_size = 1 << s
                branch_size = block_size >> 1
                for j in range(l_idx, -1, -block_size):
                    if j % block_size >= branch_size:
                        path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                        path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        llr_ch = _prepare_llr(llr_ch)
        active = [_Path(self.N, self.n)]
        active[0].L[:, 0] = llr_ch

        for phase, l_idx in enumerate(self.decode_order):
            candidates = []
            for path in active:
                self._update_llrs(path, l_idx)
                llr_val = path.L[l_idx, self.n]

                if l_idx in self.frozen_set:
                    child = self._clone(path)
                    child.u_hat[l_idx] = 0
                    child.pm += self._penalty(llr_val, 0)
                    self._update_bits(child, l_idx, 0)
                    candidates.append(child)
                else:
                    for u_bit in (0, 1):
                        child = self._clone(path)
                        child.u_hat[l_idx] = u_bit
                        child.pm += self._penalty(llr_val, u_bit)
                        self._update_bits(child, l_idx, u_bit)
                        candidates.append(child)

            candidates.sort(key=lambda p: p.pm)
            active = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p for p in active
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            pool = valid if valid else active
        else:
            pool = active

        best = min(pool, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
