"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    _bit_reversed,
    _upper_llr,
    _lower_llr,
    _active_llr_level,
    _active_bit_level,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _poly_div_gf2(dividend, divisor):
    """GF(2) 多项式长除法，返回余数"""
    dividend = list(map(int, dividend))
    divisor = list(map(int, divisor))
    while len(dividend) >= len(divisor):
        if dividend[0] == 1:
            for i in range(len(divisor)):
                dividend[i] ^= divisor[i]
        dividend.pop(0)
    return np.array(dividend, dtype=int)


def _crc_poly_bits(crc_length):
    if crc_length == 8:
        return np.array([1, 0, 0, 0, 0, 0, 1, 1, 1], dtype=int)
    return np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1], dtype=int)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_poly_bits(crc_length)
    padded = np.concatenate([info_bits, np.zeros(crc_length, dtype=int)])
    remainder = _poly_div_gf2(padded, poly)
    if len(remainder) < crc_length:
        remainder = np.concatenate([np.zeros(crc_length - len(remainder), dtype=int), remainder])
    return np.concatenate([info_bits, remainder[-crc_length:]])


def crc_check(bits, crc_length=8):
    """检验 bits 是否通过 CRC"""
    bits = np.asarray(bits, dtype=int)
    poly = _crc_poly_bits(crc_length)
    remainder = _poly_div_gf2(bits, poly)
    return len(remainder) == 0 or np.all(remainder == 0)


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length

    def _path_metric_update(self, pm, llr, bit):
        """路径度量更新"""
        if (bit == 0 and llr >= 0) or (bit == 1 and llr < 0):
            return pm
        return pm + abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n, list_size = self.N, self.n, self.list_size

        paths = [{
            'L': np.full((N, n + 1), np.nan, dtype=np.float64),
            'B': np.full((N, n + 1), np.nan),
            'pm': 0.0,
            'u': np.zeros(N, dtype=int),
        }]
        paths[0]['L'][:, 0] = llr_ch

        for phi in range(N):
            l = _bit_reversed(phi, n)
            new_paths = []

            for path in paths:
                L, B, pm, u = path['L'], path['B'], path['pm'], path['u']

                for s in range(n - _active_llr_level(l, n), n):
                    block_size = 2 ** (s + 1)
                    branch_size = block_size // 2
                    for j in range(l, N, block_size):
                        if j % block_size < branch_size:
                            L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
                        else:
                            btm_llr = L[j, s]
                            top_llr = L[j - branch_size, s]
                            top_bit = B[j - branch_size, s + 1]
                            L[j, s + 1] = _lower_llr(btm_llr, top_llr, int(top_bit))

                llr_leaf = L[l, n]

                if l in self.frozen_set:
                    u[l] = 0
                    B[l, n] = 0
                    pm_new = self._path_metric_update(pm, llr_leaf, 0)
                    new_paths.append({
                        'L': L.copy(),
                        'B': B.copy(),
                        'pm': pm_new,
                        'u': u.copy(),
                    })
                else:
                    for bit in (0, 1):
                        Lc = L.copy()
                        Bc = B.copy()
                        uc = u.copy()
                        uc[l] = bit
                        Bc[l, n] = bit
                        pm_new = self._path_metric_update(pm, llr_leaf, bit)
                        new_paths.append({
                            'L': Lc,
                            'B': Bc,
                            'pm': pm_new,
                            'u': uc,
                        })

            for path in new_paths:
                l = _bit_reversed(phi, n)
                if l >= N / 2:
                    B, u = path['B'], path['u']
                    for s in range(n, n - _active_bit_level(l, n), -1):
                        block_size = 2 ** s
                        branch_size = block_size // 2
                        for j in range(l, -1, -block_size):
                            if j % block_size >= branch_size:
                                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                                B[j, s - 1] = B[j, s]

            new_paths.sort(key=lambda p: p['pm'])
            paths = new_paths[:list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p['u'], self.crc_length)]
            if valid:
                best = min(valid, key=lambda p: p['pm'])
            else:
                best = paths[0]
        else:
            best = paths[0]

        return best['u'], best['pm']
