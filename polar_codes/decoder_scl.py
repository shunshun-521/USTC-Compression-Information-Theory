"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    f_operation,
    g_operation,
    bit_reversed_index,
    active_llr_level,
    active_bit_level,
)


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07, CRC-16: 0x8005
    """
    info_bits = np.asarray(info_bits, dtype=np.int8)
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

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确。"""
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=np.int8)
    return np.array_equal(crc_encode(bits[:-crc_length], crc_length)[-crc_length:], bits[-crc_length:])


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.llr_perm = bit_reversal_permutation(N)

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)[self.llr_perm]

        paths = [{
            'L': np.zeros((self.N, self.n + 1), dtype=np.float64),
            'B': np.zeros((self.N, self.n + 1), dtype=np.int8),
            'pm': 0.0,
            'u_hat': np.zeros(self.N, dtype=np.int8),
        }]
        paths[0]['L'][:, 0] = llr_ch

        for phi in range(self.N):
            l = bit_reversed_index(phi, self.n)
            candidates = []

            for path in paths:
                L, B = path['L'], path['B']

                for s in range(self.n - active_llr_level(l, self.n), self.n):
                    block_size = 2 ** (s + 1)
                    branch_size = block_size // 2
                    for j in range(l, self.N, block_size):
                        if j % block_size < branch_size:
                            L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                        else:
                            L[j, s + 1] = g_operation(
                                L[j - branch_size, s],
                                L[j, s],
                                B[j - branch_size, s + 1],
                            )

                llr = L[l, self.n]
                if self.frozen_bits[l]:
                    new_path = {
                        'L': L.copy(),
                        'B': B.copy(),
                        'pm': path['pm'] + self._pm_penalty(llr, 0),
                        'u_hat': path['u_hat'].copy(),
                    }
                    new_path['u_hat'][l] = 0
                    new_path['B'][l, self.n] = 0
                    self._update_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = {
                            'L': L.copy(),
                            'B': B.copy(),
                            'pm': path['pm'] + self._pm_penalty(llr, bit),
                            'u_hat': path['u_hat'].copy(),
                        }
                        new_path['u_hat'][l] = bit
                        new_path['B'][l, self.n] = bit
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p['pm'])
            paths = candidates[: self.list_size]

        crc_pass = []
        for path in paths:
            if self.crc_length > 0:
                info_bits = path['u_hat'][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_pass.append(path)
            else:
                crc_pass.append(path)

        best = min(crc_pass if crc_pass else paths, key=lambda p: p['pm'])
        return best['u_hat'].astype(int), best['pm']

    def _update_bits(self, path, l):
        B = path['B']
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = (B[j, s] + B[j - branch_size, s]) % 2
                    B[j, s - 1] = B[j, s]
