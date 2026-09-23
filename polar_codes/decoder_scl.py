"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    f_operation,
    g_operation,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    _update_llrs,
)
from encoder import bit_reversal_permutation


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_poly(crc_length):
    if crc_length == 8:
        return CRC8_POLY
    if crc_length == 16:
        return CRC16_POLY
    raise ValueError(f"Unsupported CRC length: {crc_length}")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        msb = (reg >> (crc_length - 1)) & 1
        reg = ((reg << 1) | int(bit)) & ((1 << crc_length) - 1)
        if msb ^ int(bit):
            reg ^= poly
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    if crc_length == 0:
        return True
    encoded = crc_encode(bits[:-crc_length], crc_length)
    return bool(np.array_equal(encoded, bits))


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_mask = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_mask)[0]
        self.br = bit_reversal_permutation(N)

    def _path_metric_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return abs(llr) if bit != hard else 0.0

    def _update_bits_path(self, B, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_init = llr_ch[self.br]

        paths = [{
            'L': np.zeros((self.N, self.n + 1), dtype=np.float64),
            'B': np.zeros((self.N, self.n + 1), dtype=np.int8),
            'pm': 0.0,
            'u': np.zeros(self.N, dtype=np.int8),
        }]
        paths[0]['L'][:, 0] = llr_init

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            candidates = []

            for path in paths:
                _update_llrs(path['L'], path['B'], l, self.n)
                llr0 = path['L'][l, self.n]

                if self.frozen_mask[l]:
                    new_path = {
                        'L': path['L'].copy(),
                        'B': path['B'].copy(),
                        'pm': path['pm'] + self._path_metric_penalty(llr0, 0),
                        'u': path['u'].copy(),
                    }
                    new_path['B'][l, self.n] = 0
                    new_path['u'][l] = 0
                    self._update_bits_path(new_path['B'], l)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = {
                            'L': path['L'].copy(),
                            'B': path['B'].copy(),
                            'pm': path['pm'] + self._path_metric_penalty(llr0, bit),
                            'u': path['u'].copy(),
                        }
                        new_path['B'][l, self.n] = bit
                        new_path['u'][l] = bit
                        self._update_bits_path(new_path['B'], l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p['pm'])
            paths = candidates[: self.list_size]

        best_crc = None
        best_all = paths[0]
        for path in paths:
            if self.crc_length > 0:
                info_bits = path['u'][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    if best_crc is None or path['pm'] < best_crc['pm']:
                        best_crc = path
            if path['pm'] < best_all['pm']:
                best_all = path

        chosen = best_crc if best_crc is not None else best_all
        return chosen['u'], chosen['pm']
