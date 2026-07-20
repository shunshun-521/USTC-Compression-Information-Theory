"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    _prepare_llr, _frozen_set, _bit_reversed,
    _active_llr_level, _active_bit_level, _f_exact, g_operation,
)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.uint8)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)

    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=int)
    encoded = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(encoded, bits)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits)
        self.frozen = _frozen_set(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits.astype(int) == 0)[0]

    def _penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def _update_llrs(self, L, B, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _f_exact(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1])

    def _update_bits(self, B, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        llr_perm, _ = _prepare_llr(llr_ch)

        paths = [{
            'L': np.zeros((self.N, self.n + 1), dtype=np.float64),
            'B': np.zeros((self.N, self.n + 1), dtype=np.int8),
            'pm': 0.0,
        }]
        paths[0]['L'][:, 0] = llr_perm

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            new_candidates = []

            for pidx, path in enumerate(paths):
                self._update_llrs(path['L'], path['B'], l)
                llr = path['L'][l, self.n]

                if l in self.frozen:
                    pm = path['pm'] + self._penalty(llr, 0)
                    new_candidates.append((pm, pidx, 0, True))
                else:
                    for bit in (0, 1):
                        pm = path['pm'] + self._penalty(llr, bit)
                        new_candidates.append((pm, pidx, bit, False))

            new_candidates.sort(key=lambda x: x[0])
            selected = new_candidates[:self.list_size]

            next_paths = []
            for pm, pidx, bit, is_frozen in selected:
                parent = paths[pidx]
                if is_frozen and bit == 0 and pm == parent['pm'] + self._penalty(parent['L'][l, self.n], 0):
                    child = parent
                    child['pm'] = pm
                    child['B'][l, self.n] = 0
                    self._update_bits(child['B'], l)
                else:
                    child = {
                        'L': parent['L'].copy(),
                        'B': parent['B'].copy(),
                        'pm': pm,
                    }
                    child['B'][l, self.n] = 0 if is_frozen else bit
                    self._update_bits(child['B'], l)
                next_paths.append(child)

            paths = next_paths

        best_pm = float('inf')
        best_u = paths[0]['B'][:, self.n].astype(int)
        crc_valid = []

        for path in paths:
            u_hat = path['B'][:, self.n].astype(int)
            if self.crc_length > 0:
                payload = u_hat[self.info_indices]
                if crc_check(payload, self.crc_length):
                    crc_valid.append((path['pm'], u_hat))
            if path['pm'] < best_pm:
                best_pm = path['pm']
                best_u = u_hat

        if crc_valid:
            crc_valid.sort(key=lambda x: x[0])
            return crc_valid[0][1], crc_valid[0][0]

        return best_u, best_pm
