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
    _permute_llr_for_decoder,
)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = 0x07 if crc_length == 8 else 0x8005
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length <= 0:
        return True
    bits = np.asarray(bits, dtype=int)
    return np.array_equal(bits, crc_encode(bits[:-crc_length], crc_length))


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _new_path(self, llr):
        return {
            'pm': 0.0,
            'L': np.full((self.N, self.n + 1), np.nan, dtype=np.float64),
            'B': np.full((self.N, self.n + 1), np.nan),
            'llr0': llr.copy(),
        }

    def _clone_path(self, path):
        return {
            'pm': path['pm'],
            'L': path['L'].copy(),
            'B': path['B'].copy(),
            'llr0': path['llr0'],
        }

    def _update_llrs(self, path, l):
        L, B = path['L'], path['B']
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )

    def _update_bits(self, path, l):
        B = path['B']
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        llr = _permute_llr_for_decoder(llr_ch)
        paths = [self._new_path(llr)]
        paths[0]['L'][:, 0] = llr

        for l in [_bit_reversed(i, self.n) for i in range(self.N)]:
            candidates = []
            for path in paths:
                self._update_llrs(path, l)
                llr_val = path['L'][l, self.n]
                if np.isnan(llr_val):
                    llr_val = path['L'][l, self.n - _active_llr_level(l, self.n)]
                if l in self.frozen_set:
                    pm = path['pm'] + (0.0 if llr_val >= 0 else abs(llr_val))
                    candidates.append((pm, path, 0))
                else:
                    pm0 = path['pm'] + (0.0 if llr_val >= 0 else abs(llr_val))
                    pm1 = path['pm'] + (0.0 if llr_val < 0 else abs(llr_val))
                    candidates.append((pm0, path, 0))
                    candidates.append((pm1, path, 1))

            candidates.sort(key=lambda x: x[0])
            selected = candidates[: self.list_size]

            new_paths = []
            usage = {}
            for pm, parent, bit in selected:
                count = usage.get(id(parent), 0)
                path = self._clone_path(parent) if count > 0 else parent
                usage[id(parent)] = count + 1

                path['pm'] = pm
                path['B'][l, self.n] = bit
                self._update_bits(path, l)
                new_paths.append(path)

            paths = new_paths

        if self.crc_length > 0:
            valid = []
            for path in paths:
                u_hat = path['B'][:, self.n].astype(int)
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            if valid:
                best = min(valid, key=lambda p: p['pm'])
                return best['B'][:, self.n].astype(int), best['pm']

        best = min(paths, key=lambda p: p['pm'])
        return best['B'][:, self.n].astype(int), best['pm']
