"""
极化码 SCL（串行抵消列表）译码器
Permuted SCD 顺序，支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    bit_reversed,
    _active_llr_level,
    _active_bit_level,
    _upper_llr,
    _lower_llr,
    f_operation,
)


_CRC8_GEN = [1, 0, 0, 0, 0, 1, 1, 1, 1]  # x^8 + x^2 + x + 1
_CRC16_GEN = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]


def _gf2_crc_remainder(data_bits, generator):
    reg = list(data_bits) + [0] * (len(generator) - 1)
    for i in range(len(data_bits)):
        if reg[i]:
            for j in range(len(generator)):
                reg[i + j] ^= generator[j]
    return reg[-(len(generator) - 1):]


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    gen = _CRC8_GEN if crc_length == 8 else _CRC16_GEN
    rem = _gf2_crc_remainder(info_bits.tolist(), gen)
    crc_bits = np.array(rem, dtype=np.int8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    gen = _CRC8_GEN if crc_length == 8 else _CRC16_GEN
    rem = _gf2_crc_remainder(bits.tolist(), gen)
    return all(x == 0 for x in rem)


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：分裂时复制路径）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=np.int8)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def _copy_path(self, src):
        p = _Path(self.N, self.n)
        p.L = src.L.copy()
        p.B = src.B.copy()
        p.pm = src.pm
        p.u_hat = src.u_hat.copy()
        return p

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = _upper_llr(
                        path.L[j, s], path.L[j + branch_size, s]
                    )
                else:
                    path.L[j, s + 1] = _lower_llr(
                        path.L[j, s],
                        path.L[j - branch_size, s],
                        int(path.B[j - branch_size, s + 1]),
                    )

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n)]
        paths[0].L[:, 0] = llr_ch

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                cur_llr = path.L[l, self.n]

                if self.frozen_bits[l]:
                    bit = 0
                    penalty = 0.0 if cur_llr >= 0 else abs(cur_llr)
                    new_path = self._copy_path(path)
                    new_path.pm += penalty
                    new_path.u_hat[l] = bit
                    new_path.B[l, self.n] = bit
                    self._update_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        hard = 0 if cur_llr >= 0 else 1
                        penalty = 0.0 if bit == hard else abs(cur_llr)
                        new_path = self._copy_path(path)
                        new_path.pm += penalty
                        new_path.u_hat[l] = bit
                        new_path.B[l, self.n] = bit
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[:self.list_size]

        crc_ok = []
        for p in paths:
            if self.crc_length > 0:
                payload = p.u_hat[self.info_indices]
                if crc_check(payload, self.crc_length):
                    crc_ok.append(p)
            else:
                crc_ok.append(p)

        pool = crc_ok if crc_ok else paths
        best = min(pool, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
