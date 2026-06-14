"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    _update_llr, _update_bits, f_operation, g_operation, precompute_sc_indices,
)


CRC8_GEN = [1, 0, 0, 0, 0, 0, 1, 1, 1]          # x^8 + x^2 + x + 1
CRC16_GEN = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]  # 0x8005


def _gf2_crc_encode(msg_bits, gen):
    r = len(gen) - 1
    reg = [int(b) for b in msg_bits] + [0] * r
    for i in range(len(msg_bits)):
        if reg[i]:
            for j in range(len(gen)):
                reg[i + j] ^= gen[j]
    return np.array(list(msg_bits) + reg[len(msg_bits):], dtype=np.int8)


def _gf2_crc_check(bits, gen):
    r = len(gen) - 1
    reg = [int(b) for b in bits]
    for i in range(len(bits) - r):
        if reg[i]:
            for j in range(len(gen)):
                reg[i + j] ^= gen[j]
    return all(x == 0 for x in reg[-r:])


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    gen = CRC8_GEN if crc_length == 8 else CRC16_GEN
    return _gf2_crc_encode(info_bits, gen)


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    gen = CRC8_GEN if crc_length == 8 else CRC16_GEN
    return _gf2_crc_check(bits, gen)


class SCLDecoder:
    """SCL 译码器（路径复制实现）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)

    @staticmethod
    def _pm_penalty(llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def decode(self, llr_ch):
        """主译码函数。"""
        N, n = self.N, self.n
        br = self.br
        llr0 = np.asarray(llr_ch, dtype=np.float64)[br]

        paths = [{
            'L': np.zeros((N, n + 1), dtype=np.float64),
            'B': np.zeros((N, n + 1), dtype=np.int8),
            'pm': 0.0,
            'u': np.zeros(N, dtype=np.int8),
        }]
        paths[0]['L'][:, 0] = llr0

        for i in range(N):
            l = br[i]
            candidates = []

            for path in paths:
                _update_llr(path['L'], path['B'], l, n, N)
                llr = path['L'][l, n]

                if self.frozen_bits[l]:
                    u = 0
                    new = {
                        'L': path['L'].copy(),
                        'B': path['B'].copy(),
                        'pm': path['pm'] + self._pm_penalty(llr, u),
                        'u': path['u'].copy(),
                    }
                    new['B'][l, n] = 0
                    new['u'][l] = 0
                    _update_bits(new['B'], l, n, N)
                    candidates.append(new)
                else:
                    for u in (0, 1):
                        new = {
                            'L': path['L'].copy(),
                            'B': path['B'].copy(),
                            'pm': path['pm'] + self._pm_penalty(llr, u),
                            'u': path['u'].copy(),
                        }
                        new['B'][l, n] = u
                        new['u'][l] = u
                        _update_bits(new['B'], l, n, N)
                        candidates.append(new)

            candidates.sort(key=lambda p: p['pm'])
            paths = candidates[: self.list_size]

        info_mask = ~self.frozen_bits
        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p['u'][info_mask], self.crc_length)]
            best = min(valid if valid else paths, key=lambda p: p['pm'])
        else:
            best = paths[0]

        return best['u'].astype(int), best['pm']
