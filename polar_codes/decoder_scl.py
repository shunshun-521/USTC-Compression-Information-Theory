"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversed
from decoder_sc import (
    f_operation, g_operation, precompute_sc_indices,
    _update_llrs, _update_bits, _active_llr_level, _active_bit_level,
)


_CRC8_POLY = [1, 0, 0, 0, 0, 1, 1, 1]
_CRC16_POLY = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]


def _crc_poly(bits, poly, crc_length):
    reg = list(map(int, bits))
    for i in range(len(bits) - crc_length):
        if reg[i]:
            for j in range(len(poly)):
                reg[i + j] ^= poly[j]
    return reg[-crc_length:]


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    padded = np.concatenate([info_bits, np.zeros(crc_length, dtype=int)])
    crc_bits = np.array(_crc_poly(padded, poly, crc_length), dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    if crc_length == 0:
        return True
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_poly(bits, poly, crc_length)
    return all(b == 0 for b in remainder)


class SCLDecoder:
    """SCL 译码器（Vangala 置换 SC + 路径度量）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        paths = [{
            'L': np.zeros((N, n + 1), dtype=np.float64),
            'B': np.zeros((N, n + 1), dtype=int),
            'pm': 0.0,
            'u_hat': np.zeros(N, dtype=int),
        }]
        paths[0]['L'][:, 0] = llr_ch

        for i in range(N):
            l = bit_reversed(i, n)
            candidates = []

            for path in paths:
                _update_llrs(path['L'], path['B'], l, n)
                llr = path['L'][l, n]

                if self.frozen_bits[l]:
                    new_path = {
                        'L': path['L'].copy(),
                        'B': path['B'].copy(),
                        'pm': path['pm'] + self._pm_penalty(llr, 0),
                        'u_hat': path['u_hat'].copy(),
                    }
                    new_path['B'][l, n] = 0
                    new_path['u_hat'][l] = 0
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = {
                            'L': path['L'].copy(),
                            'B': path['B'].copy(),
                            'pm': path['pm'] + self._pm_penalty(llr, bit),
                            'u_hat': path['u_hat'].copy(),
                        }
                        new_path['B'][l, n] = bit
                        new_path['u_hat'][l] = bit
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p['pm'])
            paths = candidates[: self.list_size]

            for path in paths:
                _update_bits(path['B'], l, n, N)

        best = paths[0]
        if self.crc_length > 0:
            crc_pass = [
                p for p in paths
                if crc_check(p['u_hat'][self.info_indices], self.crc_length)
            ]
            if crc_pass:
                best = min(crc_pass, key=lambda p: p['pm'])

        return best['u_hat'], best['pm']
