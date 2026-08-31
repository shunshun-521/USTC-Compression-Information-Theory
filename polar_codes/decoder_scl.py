"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    f_operation,
    g_operation,
    _active_llr_level,
    _active_bit_level,
    _bit_reversed_index,
)
from encoder import bit_reversal_permutation


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)

    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits).astype(bool)
        if self.frozen_bits.dtype != bool:
            self.frozen_bits = np.asarray(frozen_bits, dtype=int).astype(bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.rev = bit_reversal_permutation(N)
        self.decode_order = [_bit_reversed_index(i, self.n) for i in range(N)]

    def _llr_penalty(self, llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)[self.rev]

        paths = [{
            'pm': 0.0,
            'L': np.full((self.N, self.n + 1), np.nan, dtype=np.float64),
            'B': np.full((self.N, self.n + 1), np.nan, dtype=np.float64),
            'u': np.zeros(self.N, dtype=int),
        }]
        paths[0]['L'][:, 0] = llr_ch

        for phi_idx, l in enumerate(self.decode_order):
            new_paths = []
            for path in paths:
                self._update_llrs(path, l)
                cur_llr = path['L'][l, self.n]
                if np.isnan(cur_llr):
                    cur_llr = 0.0

                if self.frozen_bits[l]:
                    pen = self._llr_penalty(cur_llr, 0)
                    child = self._fork_path(path, l, 0, pen)
                    new_paths.append(child)
                else:
                    for u in (0, 1):
                        pen = self._llr_penalty(cur_llr, u)
                        child = self._fork_path(path, l, u, pen)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p['pm'])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if self._crc_ok(p['u'])]
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p['pm'])
        return best['u'], best['pm']

    def _crc_ok(self, u_hat):
        info_idx = np.where(~self.frozen_bits)[0]
        info_bits = u_hat[info_idx]
        if len(info_bits) < self.crc_length:
            return False
        return crc_check(info_bits, self.crc_length)

    def _fork_path(self, path, l, u, penalty):
        child = {
            'pm': path['pm'] + penalty,
            'L': path['L'].copy(),
            'B': path['B'].copy(),
            'u': path['u'].copy(),
        }
        child['B'][l, self.n] = u
        child['u'][l] = u
        self._update_bits(child, l)
        return child

    def _update_llrs(self, path, l):
        L = path['L']
        B = path['B']
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    if np.isnan(top_bit):
                        top_bit = 0
                    L[j, s + 1] = g_operation(L[j - branch_size, s], L[j, s], top_bit)

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        B = path['B']
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]
