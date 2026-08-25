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
)


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        if reg & top:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    crc_bits = np.zeros(crc_length, dtype=int)
    for i in range(crc_length):
        crc_bits[crc_length - 1 - i] = (reg >> i) & 1
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    data = bits[:-crc_length]
    expected = crc_encode(data, crc_length)
    return np.array_equal(bits, expected)


def _bit_reversed(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 L/B 数组）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]
        self.br_map = [_bit_reversed(i, self.n) for i in range(N)]

    def _update_llrs(self, L, B, l):
        n = self.n
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], top_bit
                    )

    def _update_bits(self, B, l):
        if l < self.N // 2:
            return
        n = self.n
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    def _path_metric_penalty(self, llr, u):
        hard = 0 if llr >= 0.0 else 1
        return 0.0 if u == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        paths = [{
            'pm': 0.0,
            'L': np.zeros((N, n + 1), dtype=np.float64),
            'B': np.zeros((N, n + 1), dtype=int),
            'u_hat': np.zeros(N, dtype=int),
        }]
        paths[0]['L'][:, 0] = llr_ch

        for phi_nat in range(N):
            l = self.br_map[phi_nat]
            candidates = []

            for path in paths:
                self._update_llrs(path['L'], path['B'], l)
                llr0 = path['L'][l, n]

                if self.frozen_bits[l]:
                    pen = self._path_metric_penalty(llr0, 0)
                    new = {
                        'pm': path['pm'] + pen,
                        'L': path['L'].copy(),
                        'B': path['B'].copy(),
                        'u_hat': path['u_hat'].copy(),
                    }
                    new['B'][l, n] = 0
                    new['u_hat'][l] = 0
                    candidates.append(new)
                else:
                    for u in (0, 1):
                        pen = self._path_metric_penalty(llr0, u)
                        new = {
                            'pm': path['pm'] + pen,
                            'L': path['L'].copy(),
                            'B': path['B'].copy(),
                            'u_hat': path['u_hat'].copy(),
                        }
                        new['B'][l, n] = u
                        new['u_hat'][l] = u
                        candidates.append(new)

            candidates.sort(key=lambda p: p['pm'])
            paths = candidates[:self.list_size]

            for path in paths:
                self._update_bits(path['B'], l)

        crc_pass = []
        if self.crc_length > 0:
            for path in paths:
                info_vals = path['u_hat'][self.info_indices]
                if crc_check(info_vals, self.crc_length):
                    crc_pass.append(path)

        best = min(crc_pass if crc_pass else paths, key=lambda p: p['pm'])
        return best['u_hat'], best['pm']
