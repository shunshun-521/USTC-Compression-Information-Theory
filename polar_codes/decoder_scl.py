"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import math
import numpy as np

from decoder_sc import (
    f_operation,
    g_operation,
    _active_llr_level,
    _active_bit_level,
    _bit_reversed,
    _update_llrs,
    _update_bits,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    mask = (1 << crc_length) - 1

    reg = 0
    for bit in np.concatenate([info_bits, np.zeros(crc_length, dtype=int)]):
        feedback = ((reg >> (crc_length - 1)) ^ int(bit)) & 1
        reg = (reg << 1) & mask
        if feedback:
            reg ^= poly

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


class SCLDecoder:
    """SCL 译码器（Lazy Copy）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [_bit_reversed(i, self.n) for i in range(N)]

    def _new_path(self, llr_ch):
        L = np.zeros((self.N, self.n + 1), dtype=np.float64)
        B = np.zeros((self.N, self.n + 1), dtype=int)
        L[:, 0] = llr_ch
        return {'L': L, 'B': B, 'pm': 0.0, 'u_hat': np.zeros(self.N, dtype=int)}

    def _pm_penalty(self, llr_val, u_bit):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr_val)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [self._new_path(llr_ch)]

        for step, l in enumerate(self.decode_order):
            candidates = []
            for path in paths:
                _update_llrs(path['L'], path['B'], l, self.n)
                llr_val = path['L'][l, self.n]

                if l in self.frozen_set:
                    path['pm'] += self._pm_penalty(llr_val, 0)
                    path['B'][l, self.n] = 0
                    path['u_hat'][l] = 0
                    _update_bits(path['B'], l, self.n, self.N)
                    candidates.append((path['pm'], path))
                else:
                    for u_bit in (0, 1):
                        new_path = {
                            'L': path['L'].copy(),
                            'B': path['B'].copy(),
                            'pm': path['pm'] + self._pm_penalty(llr_val, u_bit),
                            'u_hat': path['u_hat'].copy(),
                        }
                        new_path['B'][l, self.n] = u_bit
                        new_path['u_hat'][l] = u_bit
                        _update_bits(new_path['B'], l, self.n, self.N)
                        candidates.append((new_path['pm'], new_path))

            candidates.sort(key=lambda x: x[0])
            paths = [c[1] for c in candidates[:self.list_size]]

        if self.crc_length > 0:
            crc_pass = [p for p in paths if crc_check(p['u_hat'], self.crc_length)]
            best = min(crc_pass if crc_pass else paths, key=lambda p: p['pm'])
        else:
            best = min(paths, key=lambda p: p['pm'])

        return best['u_hat'].copy(), best['pm']
