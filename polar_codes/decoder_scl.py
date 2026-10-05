"""CRC 与 SCL 辅助"""
import math
import numpy as np
from decoder_sc import (
    bit_reversed_index,
    _active_llr_level,
    _active_bit_level,
    _upper_llr_exact,
    _lower_llr_exact,
)


def _crc_generator_bits(crc_length):
    if crc_length == 8:
        return np.array([1, 0, 0, 0, 0, 0, 1, 1, 1], dtype=int)
    if crc_length == 16:
        return np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=int)
    raise ValueError("crc_length must be 8 or 16")


def _gf2_crc_remainder(msg, gen):
    msg = msg.astype(int).tolist()
    gen = gen.astype(int).tolist()
    n = len(gen)
    for i in range(len(msg) - n + 1):
        if msg[i]:
            for j in range(n):
                msg[i + j] ^= gen[j]
    return np.array(msg[-(n - 1) :], dtype=int)


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    gen = _crc_generator_bits(crc_length)
    padded = np.concatenate([info_bits, np.zeros(crc_length, dtype=int)])
    rem = _gf2_crc_remainder(padded, gen)
    return np.concatenate([info_bits, rem])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    gen = _crc_generator_bits(crc_length)
    rem = _gf2_crc_remainder(bits, gen)
    return np.all(rem == 0)


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _llr_penalty(self, llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def _advance_path(self, path, l):
        n = self.n
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = _upper_llr_exact(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = _lower_llr_exact(
                        path.L[j, s], path.L[j - branch_size, s], path.B[j - branch_size, s + 1]
                    )

    def _propagate_bits(self, path, l, u_bit):
        n = self.n
        path.B[l, n] = u_bit
        path.u_hat[l] = u_bit
        if l < self.N // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                    path.B[j, s - 1] = path.B[j, s]

    def _clone(self, path):
        cp = _Path(self.N, self.n, path.L[:, 0])
        cp.L = path.L.copy()
        cp.B = path.B.copy()
        cp.pm = path.pm
        cp.u_hat = path.u_hat.copy()
        return cp

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch.copy())]
        decode_order = [bit_reversed_index(i, self.n) for i in range(self.N)]

        for l in decode_order:
            new_paths = []
            for path in paths:
                self._advance_path(path, l)
                llr_leaf = path.L[l, self.n]
                if l in self.frozen_set:
                    cp = self._clone(path)
                    cp.pm += self._llr_penalty(llr_leaf, 0)
                    self._propagate_bits(cp, l, 0)
                    new_paths.append(cp)
                else:
                    for u_c in (0, 1):
                        cp = self._clone(path)
                        cp.pm += self._llr_penalty(llr_leaf, u_c)
                        self._propagate_bits(cp, l, u_c)
                        new_paths.append(cp)
            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat[self.info_indices], self.crc_length)]
            chosen = min(valid, key=lambda p: p.pm) if valid else min(paths, key=lambda p: p.pm)
        else:
            chosen = min(paths, key=lambda p: p.pm)
        return chosen.u_hat.copy(), chosen.pm
