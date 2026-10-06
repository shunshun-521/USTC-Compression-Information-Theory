"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    bit_reversed,
    _active_llr_level,
    _active_bit_level,
    _upper_llr,
    _lower_llr,
)


def _crc_bits(bits, crc_length):
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")
    mask = (1 << crc_length) - 1
    reg = 0
    for b in bits:
        fb = ((reg >> (crc_length - 1)) & 1) ^ int(b)
        reg = (reg << 1) & mask
        if fb:
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    """CRC-8 (0x07) 或 CRC-16 (0x8005)"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    reg = _crc_bits(info_bits, crc_length)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    return _crc_bits(bits, crc_length) == 0


class _Path:
    __slots__ = ("L", "B", "pm", "active")

    def __init__(self, N, n, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = llr_ch.copy()
        self.pm = 0.0
        self.active = True


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径共享 LLR/B 数组，分裂时复制）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.L_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _pm_penalty(self, llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, self.n]
                if l in self.frozen_set:
                    pen = self._pm_penalty(llr, 0)
                    clone = _Path(self.N, self.n, llr_ch)
                    clone.L = path.L.copy()
                    clone.B = path.B.copy()
                    clone.pm = path.pm + pen
                    clone.B[l, self.n] = 0
                    self._update_bits(clone, l)
                    candidates.append(clone)
                else:
                    for u in (0, 1):
                        clone = _Path(self.N, self.n, llr_ch)
                        clone.L = path.L.copy()
                        clone.B = path.B.copy()
                        clone.pm = path.pm + self._pm_penalty(llr, u)
                        clone.B[l, self.n] = u
                        self._update_bits(clone, l)
                        candidates.append(clone)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.L_size]

        u_stack = [p.B[:, self.n].astype(np.int8) for p in paths]
        pms = [p.pm for p in paths]

        if self.crc_length > 0:
            valid = []
            for u_hat, pm in zip(u_stack, pms):
                payload = u_hat[self.info_indices]
                if crc_check(payload, self.crc_length):
                    valid.append((pm, u_hat))
            if valid:
                valid.sort(key=lambda x: x[0])
                return valid[0][1], valid[0][0]

        best = int(np.argmin(pms))
        return u_stack[best], pms[best]

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = _upper_llr(path.L[j, s], path.L[j + branch_size, s])
                else:
                    top_bit = path.B[j - branch_size, s + 1]
                    path.L[j, s + 1] = _lower_llr(
                        path.L[j, s], path.L[j - branch_size, s], top_bit
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = (
                        path.B[j, s] ^ path.B[j - branch_size, s]
                    ) & 1
                    path.B[j, s - 1] = path.B[j, s]
