"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL），基于 Vangala 置换 SC 框架
"""
import numpy as np
from decoder_sc import (
    upper_llr,
    lower_llr,
    active_llr_level,
    active_bit_level,
)
from encoder import bit_reversed


CRC8_POLY_BITS = np.array([1, 0, 0, 0, 0, 0, 1, 1, 1], dtype=int)
CRC16_POLY_BITS = np.array(
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=int
)


def _poly_generator(crc_length):
    if crc_length == 8:
        return CRC8_POLY_BITS
    if crc_length == 16:
        return CRC16_POLY_BITS
    raise ValueError(f"Unsupported CRC length: {crc_length}")


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    gen = _poly_generator(crc_length)
    r = crc_length
    msg = np.concatenate([info_bits, np.zeros(r, dtype=int)])
    for i in range(len(msg) - r):
        if msg[i] == 1:
            msg[i : i + len(gen)] ^= gen
    return np.concatenate([info_bits, msg[-r:]])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    gen = _poly_generator(crc_length)
    r = crc_length
    msg = bits.copy()
    for i in range(len(bits) - r):
        if msg[i] == 1:
            msg[i : i + len(gen)] ^= gen
    return np.all(msg[-r:] == 0)


def _pm_update(pm, llr, u):
    u_hard = 0 if llr >= 0 else 1
    if u != u_hard:
        pm += abs(llr)
    return pm


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)

    def copy(self):
        p = _Path(self.L.shape[0], self.L.shape[1] - 1)
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.pm = self.pm
        p.u_hat = self.u_hat.copy()
        return p


class SCLDecoder:
    """SCL 译码器（Lazy Copy）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.decode_order = [bit_reversed(i, self.n) for i in range(N)]

    def _update_llrs(self, path, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            bs = 1 << (s + 1)
            br = bs // 2
            for j in range(l, self.N, bs):
                if j % bs < br:
                    path.L[j, s + 1] = upper_llr(path.L[j, s], path.L[j + br, s])
                else:
                    path.L[j, s + 1] = lower_llr(
                        path.L[j, s],
                        path.L[j - br, s],
                        path.B[j - br, s + 1],
                    )

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            bs = 1 << s
            br = bs // 2
            for j in range(l, -1, -bs):
                if j % bs >= br:
                    path.B[j - br, s - 1] = int(path.B[j, s]) ^ int(path.B[j - br, s])
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n)]
        paths[0].L[:, 0] = llr_ch

        for l in self.decode_order:
            for path in paths:
                self._update_llrs(path, l)

            if l in self.frozen_set:
                for path in paths:
                    path.pm = _pm_update(path.pm, path.L[l, self.n], 0)
                    path.B[l, self.n] = 0
                    path.u_hat[l] = 0
                    self._update_bits(path, l)
            else:
                candidates = []
                for path in paths:
                    llr_val = path.L[l, self.n]
                    for u in (0, 1):
                        child = path.copy()
                        child.pm = _pm_update(child.pm, llr_val, u)
                        child.B[l, self.n] = u
                        child.u_hat[l] = u
                        self._update_bits(child, l)
                        candidates.append(child)
                candidates.sort(key=lambda p: p.pm)
                paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            best = min(valid if valid else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
