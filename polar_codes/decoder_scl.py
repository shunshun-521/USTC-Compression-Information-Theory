"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math

from encoder import bit_reversal_permutation
from decoder_sc import (
    f_operation, g_operation,
    _bit_reversed, _active_llr_level, _active_bit_level,
    _update_llrs, _update_bits,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_checksum(info_bits, crc_length):
    """基于 LFSR 的 CRC 校验和（多项式 0x07 / 0x8005）。"""
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in info_bits:
        reg ^= int(bit)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    reg = _crc_checksum(info_bits, crc_length)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 是否通过 CRC 校验。"""
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits, expected)


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)

    def _pm_update(self, pm, llr, u):
        penalty = 0.0 if (u == 0 and llr >= 0) or (u == 1 and llr < 0) else abs(llr)
        return pm + penalty

    def decode(self, llr_ch):
        """主译码函数。返回: (u_hat, pm)"""
        N = self.N
        n = self.n
        L_size = self.list_size

        paths = [{
            'pm': 0.0,
            'L': np.zeros((N, n + 1), dtype=np.float64),
            'B': np.zeros((N, n + 1), dtype=int),
            'u_hat': np.zeros(N, dtype=int),
        }]
        paths[0]['L'][:, 0] = llr_ch[self.br]

        for i in range(N):
            l = _bit_reversed(i, n)
            new_paths = []

            for path in paths:
                L, B = path['L'], path['B']
                _update_llrs(L, B, l, n)
                cur_llr = L[l, n]

                if l in self.frozen_set:
                    new_paths.append({
                        'pm': self._pm_update(path['pm'], cur_llr, 0),
                        'L': L.copy(),
                        'B': B.copy(),
                        'u_hat': path['u_hat'].copy(),
                    })
                    new_paths[-1]['u_hat'][l] = 0
                    new_paths[-1]['B'][l, n] = 0
                else:
                    for u in (0, 1):
                        new_paths.append({
                            'pm': self._pm_update(path['pm'], cur_llr, u),
                            'L': L.copy(),
                            'B': B.copy(),
                            'u_hat': path['u_hat'].copy(),
                        })
                        new_paths[-1]['u_hat'][l] = u
                        new_paths[-1]['B'][l, n] = u

            for p in new_paths:
                _update_bits(p['B'], l, n, N)

            new_paths.sort(key=lambda x: x['pm'])
            paths = new_paths[:L_size]

        if self.crc_length > 0:
            info_mask = self.frozen_bits == 0
            valid = [p for p in paths if crc_check(p['u_hat'][info_mask], self.crc_length)]
            best = min(valid if valid else paths, key=lambda x: x['pm'])
        else:
            best = min(paths, key=lambda x: x['pm'])

        return best['u_hat'], best['pm']
