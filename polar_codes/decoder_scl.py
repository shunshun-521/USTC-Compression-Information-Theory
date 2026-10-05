"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _hard_decision,
    _lower_llr,
    _update_bits,
    _upper_llr,
    f_operation,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8 if crc_length == 8 else 1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = [(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)]
    return np.concatenate([info_bits, np.array(crc_bits, dtype=np.int8)])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    if len(bits) < crc_length:
        return False
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(expected, bits)


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat", "active")

    def __init__(self, N, n, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.active = True


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 P/C）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)

    def _update_llrs_path(self, path, l, f_fn):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size >> 1
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = _upper_llr(
                        path.L[j, s], path.L[j + branch_size, s], f_fn
                    )
                else:
                    top_bit = path.B[j - branch_size, s + 1]
                    path.L[j, s + 1] = _lower_llr(
                        path.L[j, s], path.L[j - branch_size, s], top_bit
                    )

    def _update_bits_path(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size >> 1
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch, f_fn=None):
        if f_fn is None:
            f_fn = f_operation
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = int(self.br[i])
            candidates = []

            for path in paths:
                if not path.active:
                    continue
                self._update_llrs_path(path, l, f_fn)
                llr_bit = path.L[l, self.n]
                if np.isnan(llr_bit):
                    llr_bit = path.L[l, self.n] = 0.0

                if l in self.frozen_set:
                    pen = 0.0 if llr_bit >= 0 else abs(llr_bit)
                    new_path = _Path(self.N, self.n, llr_ch)
                    new_path.L = path.L.copy()
                    new_path.B = path.B.copy()
                    new_path.pm = path.pm + pen
                    new_path.u_hat = path.u_hat.copy()
                    new_path.u_hat[l] = 0
                    new_path.B[l, self.n] = 0
                    self._update_bits_path(new_path, l)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        pen = 0.0 if (llr_bit >= 0 and bit == 0) or (llr_bit < 0 and bit == 1) else abs(llr_bit)
                        new_path = _Path(self.N, self.n, llr_ch)
                        new_path.L = path.L.copy()
                        new_path.B = path.B.copy()
                        new_path.pm = path.pm + pen
                        new_path.u_hat = path.u_hat.copy()
                        new_path.u_hat[l] = bit
                        new_path.B[l, self.n] = bit
                        self._update_bits_path(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]
            if not paths:
                paths = [_Path(self.N, self.n, llr_ch)]

        best = min(paths, key=lambda p: p.pm)
        if self.crc_length > 0:
            info_idx = np.where(self.frozen_bits == 0)[0]
            valid = [
                p
                for p in paths
                if crc_check(p.u_hat[info_idx], self.crc_length)
            ]
            if valid:
                best = min(valid, key=lambda p: p.pm)
        u_hat = best.u_hat.copy()
        for idx in self.frozen_set:
            u_hat[idx] = 0
        return u_hat, best.pm
