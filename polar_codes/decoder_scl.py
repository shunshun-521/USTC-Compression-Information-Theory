"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    sc_decode,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """CRC 校验位附加在信息比特之后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_poly(crc_length)
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    for b in info_bits:
        feedback = ((reg >> (crc_length - 1)) & 1) ^ int(b)
        reg = ((reg << 1) & mask) ^ (poly if feedback else 0)
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    poly = _crc_poly(crc_length)
    reg = 0
    mask = (1 << crc_length) - 1
    for b in bits:
        feedback = ((reg >> (crc_length - 1)) & 1) ^ int(b)
        reg = ((reg << 1) & mask) ^ (poly if feedback else 0)
    return reg == 0


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n, llr):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（路径复制实现）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_idx = set(np.where(self.frozen_bits.astype(bool))[0].tolist())
        self.list_size = list_size
        self.crc_length = crc_length

    def _advance_llr(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    top_bit = int(path.B[j - branch_size, s + 1])
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s],
                        path.L[j, s],
                        top_bit,
                    )

    def _propagate_bits(self, path, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if hard == bit else abs(llr)

    def decode(self, llr_ch):
        from encoder import bit_reversal_permutation

        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        br = bit_reversal_permutation(self.N)
        llr = np.asarray(llr_ch, dtype=np.float64)[br]
        paths = [_Path(self.N, self.n, llr)]

        for phase in range(self.N):
            l = _bit_reversed(phase, self.n)
            candidates = []

            for path in paths:
                self._advance_llr(path, l)
                cur_llr = path.L[l, self.n]

                if l in self.frozen_idx:
                    bit = 0
                    new_pm = path.pm + self._pm_penalty(cur_llr, bit)
                    path.B[l, self.n] = 0
                    path.u_hat[l] = 0
                    self._propagate_bits(path, l)
                    candidates.append((new_pm, path))
                else:
                    for bit in (0, 1):
                        new_path = _Path(self.N, self.n, llr)
                        new_path.L = path.L.copy()
                        new_path.B = path.B.copy()
                        new_path.u_hat = path.u_hat.copy()
                        new_path.pm = path.pm + self._pm_penalty(cur_llr, bit)
                        new_path.B[l, self.n] = bit
                        new_path.u_hat[l] = bit
                        self._propagate_bits(new_path, l)
                        candidates.append((new_path.pm, new_path))

            candidates.sort(key=lambda x: x[0])
            paths = [p for _, p in candidates[: self.list_size]]

        best = min(paths, key=lambda p: p.pm)
        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p.u_hat[np.where(self.frozen_bits == 0)[0]], self.crc_length)
            ]
            if valid:
                best = min(valid, key=lambda p: p.pm)
        return best.u_hat.copy(), float(best.pm)
