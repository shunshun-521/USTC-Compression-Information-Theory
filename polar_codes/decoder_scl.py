"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    f_operation,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    _lower_llr,
    sc_decode,
)


CRC8_POLY_BITS = np.array([1, 0, 0, 0, 0, 0, 1, 1, 1], dtype=np.int8)
CRC16_POLY_BITS = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1], dtype=np.int8)


def _gf2_remainder(dividend, divisor):
    dividend = list(map(int, dividend))
    divisor = list(map(int, divisor))
    while len(dividend) >= len(divisor):
        if dividend[0] == 1:
            for i in range(len(divisor)):
                dividend[i] ^= divisor[i]
        dividend.pop(0)
    return np.array(dividend, dtype=np.int8)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY_BITS if crc_length == 8 else CRC16_POLY_BITS
    msg = np.concatenate([info_bits, np.zeros(crc_length, dtype=np.int8)])
    rem = _gf2_remainder(msg, poly)
    if len(rem) < crc_length:
        rem = np.concatenate([np.zeros(crc_length - len(rem), dtype=np.int8), rem])
    return np.concatenate([info_bits, rem[-crc_length:]])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = CRC8_POLY_BITS if crc_length == 8 else CRC16_POLY_BITS
    rem = _gf2_remainder(bits, poly)
    return len(rem) == 0 or np.all(rem == 0)


def _path_metric_penalty(llr, bit):
    """路径度量惩罚：与 LLR 符号不一致时加 |LLR|。"""
    hard = 0 if llr >= 0 else 1
    return 0.0 if hard == bit else abs(llr)


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)。"""
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        paths = [{
            'L': np.full((N, n + 1), np.nan, dtype=np.float64),
            'B': np.full((N, n + 1), np.nan, dtype=np.float64),
            'pm': 0.0,
        }]
        paths[0]['L'][:, 0] = llr_ch

        for i in range(N):
            l = _bit_reversed(i, n)
            new_paths = []

            for path in paths:
                self._update_llrs(path, l)
                llr_leaf = path['L'][l, n]

                if l in self.frozen_set:
                    pm = path['pm'] + _path_metric_penalty(llr_leaf, 0)
                    p2 = self._copy_path(path)
                    p2['pm'] = pm
                    p2['B'][l, n] = 0
                    self._update_bits(p2, l)
                    new_paths.append(p2)
                else:
                    for bit in (0, 1):
                        pm = path['pm'] + _path_metric_penalty(llr_leaf, bit)
                        p2 = self._copy_path(path)
                        p2['pm'] = pm
                        p2['B'][l, n] = bit
                        self._update_bits(p2, l)
                        new_paths.append(p2)

            new_paths.sort(key=lambda p: p['pm'])
            paths = new_paths[: self.list_size]

        paths.sort(key=lambda p: p['pm'])
        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p['B'][:, n].astype(np.int8), self.crc_length)]
            if valid:
                paths = valid

        best = paths[0]
        return best['B'][:, n].astype(np.int8), best['pm']

    def _copy_path(self, path):
        return {'L': path['L'].copy(), 'B': path['B'].copy(), 'pm': path['pm']}

    def _update_llrs(self, path, l):
        L, B = path['L'], path['B']
        n = self.n
        N = self.N
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = int(B[j - branch_size, s + 1])
                    L[j, s + 1] = _lower_llr(L[j, s], L[j - branch_size, s], top_bit)

    def _update_bits(self, path, l):
        B = path['B']
        n = self.n
        N = self.N
        if l < N // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]
