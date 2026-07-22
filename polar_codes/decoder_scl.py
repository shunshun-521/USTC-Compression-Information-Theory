"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    _bit_reversed, _active_llr_level, _active_bit_level,
    _update_llrs, _update_bits, f_operation, g_operation,
)


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    mask = (1 << crc_length) - 1

    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=np.int8)
    return np.array_equal(bits, crc_encode(bits[:-crc_length], crc_length))


class SCLDecoder:
    """SCL 译码器（Vangala Permuted SCD + Lazy Copy）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(np.asarray(frozen_bits, dtype=bool))[0])
        self.info_mask = ~np.asarray(frozen_bits, dtype=bool)

    def _pm_penalty(self, llr, u_bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr)

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)"""
        N = self.N
        n = self.n
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        # 路径: dict with pm, L, B
        paths = [{
            'pm': 0.0,
            'L': np.full((N, n + 1), np.nan, dtype=np.float64),
            'B': np.full((N, n + 1), np.nan),
        }]
        paths[0]['L'][:, 0] = llr_ch

        for i in range(N):
            l = _bit_reversed(i, n)
            new_paths = []

            for path in paths:
                L, B = path['L'], path['B']
                _update_llrs(L, B, l, n)
                llr = L[l, n]

                if l in self.frozen_set:
                    pm = path['pm'] + self._pm_penalty(llr, 0)
                    Lc = L.copy()
                    Bc = B.copy()
                    Bc[l, n] = 0
                    _update_bits(Bc, l, n, N)
                    new_paths.append({'pm': pm, 'L': Lc, 'B': Bc})
                else:
                    for u_bit in (0, 1):
                        pm = path['pm'] + self._pm_penalty(llr, u_bit)
                        Lc = L.copy()
                        Bc = B.copy()
                        Bc[l, n] = u_bit
                        _update_bits(Bc, l, n, N)
                        new_paths.append({'pm': pm, 'L': Lc, 'B': Bc})

            new_paths.sort(key=lambda p: p['pm'])
            paths = new_paths[:self.list_size]

        # 选择最优路径
        best_path = None
        best_pm = float('inf')

        if self.crc_length > 0:
            crc_paths = []
            for path in paths:
                u_hat = path['B'][:, n].astype(int)
                info_bits = u_hat[self.info_mask]
                if crc_check(info_bits, self.crc_length):
                    crc_paths.append(path)
            if crc_paths:
                crc_paths.sort(key=lambda p: p['pm'])
                best_path = crc_paths[0]

        if best_path is None:
            paths.sort(key=lambda p: p['pm'])
            best_path = paths[0]

        u_hat = best_path['B'][:, n].astype(int)
        return u_hat, best_path['pm']
