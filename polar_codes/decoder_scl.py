"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    bit_reversed,
    active_llr_level,
    active_bit_level,
    upper_llr_min_sum,
    lower_llr_min_sum,
    channel_llr_to_decoder,
)


def crc_encode(info_bits, crc_length=8):
    """CRC-8 (0x07) 或 CRC-16 (0x8005)。"""
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")
    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array([(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    return np.array_equal(crc_encode(bits[:-crc_length], crc_length), bits)


def _update_llrs(path, l, n, N):
    for s in range(n - active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                path.L[j, s + 1] = upper_llr_min_sum(path.L[j, s], path.L[j + branch_size, s])
            else:
                top_bit = int(path.B[j - branch_size, s + 1])
                path.L[j, s + 1] = lower_llr_min_sum(path.L[j, s], path.L[j - branch_size, s], top_bit)


def _update_bits(path, l, n, N):
    if l < N / 2:
        return
    for s in range(n, n - active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                path.B[j, s - 1] = path.B[j, s]


class SCLDecoder:
    """SCL 译码器（路径复制）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        frozen_bits = np.asarray(frozen_bits)
        if frozen_bits.dtype == bool:
            self.frozen = set(np.where(frozen_bits)[0])
        else:
            self.frozen = set(np.where(frozen_bits.astype(int) != 0)[0])
        self.L_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.array(sorted(set(range(N)) - self.frozen), dtype=int)

    def _new_path(self, llr0):
        path = type("Path", (), {})()
        path.L = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        path.B = np.full((self.N, self.n + 1), np.nan)
        path.L[:, 0] = llr0
        path.pm = 0.0
        path.u_hat = np.zeros(self.N, dtype=int)
        return path

    @staticmethod
    def _pm_penalty(llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr0 = channel_llr_to_decoder(llr_ch)
        paths = [self._new_path(llr0)]

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            candidates = []
            for pidx, path in enumerate(paths):
                _update_llrs(path, l, self.n, self.N)
                llr = path.L[l, self.n]
                if l in self.frozen:
                    candidates.append((path.pm + self._pm_penalty(llr, 0), pidx, 0))
                else:
                    for bit in (0, 1):
                        candidates.append(
                            (path.pm + self._pm_penalty(llr, bit), pidx, bit)
                        )

            candidates.sort(key=lambda x: x[0])
            new_paths = []
            for pm, pidx, bit in candidates[: self.L_size]:
                src = paths[pidx]
                path = self._new_path(llr0)
                path.L = src.L.copy()
                path.B = src.B.copy()
                path.u_hat = src.u_hat.copy()
                path.pm = pm
                path.B[l, self.n] = bit
                path.u_hat[l] = bit
                _update_bits(path, l, self.n, self.N)
                new_paths.append(path)
            paths = new_paths

        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            if valid:
                paths = valid
        best = min(paths, key=lambda p: p.pm)
        return best.u_hat, best.pm
