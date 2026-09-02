"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from channel import permute_llr_for_decode
from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _lower_llr,
    _upper_llr,
)


CRC8_POLY_BITS = [1, 0, 0, 0, 0, 0, 1, 1, 1]
CRC16_POLY_BITS = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]


def _poly_bits(crc_length):
    if crc_length == 8:
        return CRC8_POLY_BITS
    if crc_length == 16:
        return CRC16_POLY_BITS
    raise ValueError(f"Unsupported CRC length: {crc_length}")


def _gf2_remainder(msg_bits, gen_bits):
    msg = [int(b) for b in msg_bits]
    n = len(gen_bits) - 1
    for i in range(len(msg) - n):
        if msg[i] == 1:
            for j in range(len(gen_bits)):
                msg[i + j] ^= gen_bits[j]
    return np.array(msg[-n:], dtype=np.int8)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    gen = _poly_bits(crc_length)
    remainder = _gf2_remainder(
        np.concatenate([info_bits, np.zeros(crc_length, dtype=np.int8)]), gen
    )
    return np.concatenate([info_bits, remainder])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    gen = _poly_bits(crc_length)
    remainder = _gf2_remainder(bits, gen)
    return np.all(remainder == 0)


class SCLDecoder:
    """SCL 译码器（Lazy Copy）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=np.int8)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def _path_metric_penalty(self, llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def _init_path(self, llr_ch):
        L = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        B = np.full((self.N, self.n + 1), np.nan)
        L[:, 0] = permute_llr_for_decode(llr_ch)
        return {'L': L, 'B': B, 'pm': 0.0, 'u_hat': np.zeros(self.N, dtype=np.int8)}

    def _copy_path(self, path):
        return {
            'L': path['L'].copy(),
            'B': path['B'].copy(),
            'pm': path['pm'],
            'u_hat': path['u_hat'].copy(),
        }

    def _update_llrs(self, path, l):
        L, B = path['L'], path['B']
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = _lower_llr(
                        L[j, s],
                        L[j - branch_size, s],
                        int(B[j - branch_size, s + 1]),
                    )

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        L, B = path['L'], path['B']
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [self._init_path(llr_ch)]

        for l in [_bit_reversed(i, self.n) for i in range(self.N)]:
            candidates = []
            for path in paths:
                self._update_llrs(path, l)
                llr = path['L'][l, self.n]

                if l in self.frozen_set:
                    new_path = self._copy_path(path)
                    new_path['pm'] += self._path_metric_penalty(llr, 0)
                    new_path['u_hat'][l] = 0
                    new_path['B'][l, self.n] = 0
                    self._update_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for u in (0, 1):
                        new_path = self._copy_path(path)
                        new_path['pm'] += self._path_metric_penalty(llr, u)
                        new_path['u_hat'][l] = u
                        new_path['B'][l, self.n] = u
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p['pm'])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for path in paths:
                info_bits = path['u_hat'][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            pool = valid if valid else paths
        else:
            pool = paths

        best = min(pool, key=lambda p: p['pm'])
        return best['u_hat'].copy(), best['pm']
