"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    sc_decode,
    upper_llr,
    lower_llr,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    if len(bits) < crc_length:
        return False
    poly = _crc_poly(crc_length)
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


def _pm_update(pm, llr, u):
    u_hard = 0 if llr >= 0 else 1
    if u != u_hard:
        return pm + abs(llr)
    return pm


class _Path:
    __slots__ = ("pm", "L", "B")

    def __init__(self, N, n, llr_ch, pm=0.0):
        self.pm = pm
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch


def _update_llrs(path, l, n, N):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                path.L[j, s + 1] = upper_llr(path.L[j, s], path.L[j + branch_size, s])
            else:
                path.L[j, s + 1] = lower_llr(
                    path.L[j, s],
                    path.L[j - branch_size, s],
                    int(path.B[j - branch_size, s + 1]),
                )


def _update_bits(path, l, n, N):
    if l < N // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                    path.B[j - branch_size, s]
                )
                path.B[j, s - 1] = path.B[j, s]


class SCLDecoder:
    """Permuted SCL 译码器（Lazy Copy：分裂时复制 L/B）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        N, n = self.N, self.n
        paths = [_Path(N, n, llr_ch)]

        for l in [_bit_reversed(i, n) for i in range(N)]:
            candidates = []
            for pidx, path in enumerate(paths):
                _update_llrs(path, l, n, N)
                llr0 = path.L[l, n]
                if l in self.frozen_set:
                    candidates.append((_pm_update(path.pm, llr0, 0), pidx, 0))
                else:
                    for u_cand in (0, 1):
                        candidates.append((_pm_update(path.pm, llr0, u_cand), pidx, u_cand))

            candidates.sort(key=lambda x: x[0])
            candidates = candidates[: self.list_size]

            new_paths = []
            for pm_new, pidx, u_bit in candidates:
                parent = paths[pidx]
                child = _Path(N, n, llr_ch, pm_new)
                child.L = parent.L.copy()
                child.B = parent.B.copy()
                child.B[l, n] = 0 if l in self.frozen_set else u_bit
                _update_bits(child, l, n, N)
                new_paths.append(child)
            paths = new_paths

        if self.crc_length > 0:
            valid = []
            for path in paths:
                u_hat = path.B[:, n].astype(np.int8)
                u_hat[self.frozen_bits] = 0
                info_bits = u_hat[~self.frozen_bits]
                if crc_check(info_bits, self.crc_length):
                    valid.append((path.pm, path))
            if valid:
                valid.sort(key=lambda x: x[0])
                best = valid[0][1]
            else:
                best = min(paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        u_hat = best.B[:, n].astype(np.int8)
        u_hat[self.frozen_bits] = 0
        return u_hat, best.pm
