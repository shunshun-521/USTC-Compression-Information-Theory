"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    f_operation,
    g_operation,
    _active_llr_level,
    _active_bit_level,
    path_metric_update,
)
from encoder import bit_reversal_permutation


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _generator_poly(crc_length):
    if crc_length == 8:
        return np.array([1, 0, 0, 0, 0, 0, 1, 1, 1], dtype=int)
    return np.array(
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1], dtype=int
    )


def _crc_remainder(bits, crc_length):
    gen = _generator_poly(crc_length)
    msg = np.concatenate([np.asarray(bits, dtype=int), np.zeros(crc_length, dtype=int)])
    for i in range(len(bits)):
        if msg[i] == 1:
            msg[i : i + len(gen)] = (msg[i : i + len(gen)] + gen) % 2
    return msg[-crc_length:]


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    crc_bits = _crc_remainder(info_bits, crc_length)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    rem = _crc_remainder(bits[:-crc_length], crc_length)
    return np.array_equal(rem, bits[-crc_length:])


class _Path:
    __slots__ = ("pm", "u_hat", "L", "B")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.L[:, 0] = llr_ch

    def copy(self):
        p = _Path.__new__(_Path)
        p.pm = self.pm
        p.u_hat = self.u_hat.copy()
        p.L = self.L.copy()
        p.B = self.B.copy()
        return p


def _path_update_llrs(path, l, n):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        N = path.L.shape[0]
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
            else:
                path.L[j, s + 1] = g_operation(
                    path.L[j - branch_size, s],
                    path.L[j, s],
                    path.B[j - branch_size, s + 1],
                )


def _path_update_bits(path, l, n):
    N = path.B.shape[0]
    if l < N // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                    path.B[j - branch_size, s]
                )
                path.B[j, s - 1] = path.B[j, s]


class SCLDecoder:
    """SCL 译码器（Permuted SC + Lazy Copy）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.rev = bit_reversal_permutation(N)
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])

        if crc_length > 0:
            info_positions = np.where(self.frozen_bits == 0)[0]
            self.crc_positions = info_positions[-crc_length:]
            self.payload_positions = info_positions[:-crc_length]
        else:
            self.crc_positions = np.array([], dtype=int)
            self.payload_positions = np.where(self.frozen_bits == 0)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = self.rev[i]
            is_frozen = l in self.frozen_set
            new_paths = []

            for path in paths:
                _path_update_llrs(path, l, self.n)
                llr = path.L[l, self.n]

                if is_frozen:
                    p = path.copy()
                    p.pm = path_metric_update(p.pm, llr, 0)
                    p.B[l, self.n] = 0
                    p.u_hat[l] = 0
                    _path_update_bits(p, l, self.n)
                    new_paths.append(p)
                else:
                    for u in (0, 1):
                        p = path.copy()
                        p.pm = path_metric_update(p.pm, llr, u)
                        p.B[l, self.n] = u
                        p.u_hat[l] = u
                        _path_update_bits(p, l, self.n)
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p.u_hat[self.payload_positions]
                crc_bits = p.u_hat[self.crc_positions]
                if crc_check(np.concatenate([info_bits, crc_bits]), self.crc_length):
                    valid.append(p)
            best = min(valid, key=lambda p: p.pm) if valid else paths[0]
        else:
            best = paths[0]

        return best.u_hat.copy(), best.pm
