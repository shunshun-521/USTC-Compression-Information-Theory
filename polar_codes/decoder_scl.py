"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation
from decoder_sc import (
    f_operation, g_operation, sc_decode,
    _bit_reversed, _active_llr_level, _active_bit_level,
    _update_llrs, _update_bits,
)


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, crc_length):
    """MSB-first CRC remainder."""
    bits = np.asarray(bits, dtype=int).tolist()
    if crc_length == 8:
        poly = _CRC8_POLY
        width = 8
    elif crc_length == 16:
        poly = _CRC16_POLY
        width = 16
    else:
        raise ValueError(f'Unsupported CRC length: {crc_length}')

    reg = 0
    for b in bits:
        reg ^= (b << (width - 1))
        for _ in range(width):
            if reg & (1 << (width - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << width) - 1)
            else:
                reg = (reg << 1) & ((1 << width) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    remainder = _crc_remainder(info_bits, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    remainder = _crc_remainder(bits[:-crc_length], crc_length)
    expected = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.array_equal(bits[-crc_length:], expected)


# ==================== SCL 译码器 ====================

class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self._inv_br = np.argsort(bit_reversal_permutation(N))

    def _pm_update(self, pm, llr, u):
        hard = 0 if llr >= 0 else 1
        if u != hard:
            pm += abs(llr)
        return pm

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)。"""
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        N, n = self.N, self.n
        llr_internal = np.asarray(llr_ch, dtype=np.float64)[self._inv_br]

        paths = [{
            'L': np.zeros((N, n + 1)),
            'B': np.zeros((N, n + 1), dtype=int),
            'pm': 0.0,
            'u_hat': np.zeros(N, dtype=int),
        }]
        paths[0]['L'][:, 0] = llr_internal

        for i in range(N):
            l = _bit_reversed(i, n)
            new_paths = []

            for path in paths:
                L, B, pm, u_hat = path['L'], path['B'], path['pm'], path['u_hat']
                _update_llrs(L, B, l, n, N)
                llr = L[l, n]

                if l in self.frozen_set:
                    u = 0
                    pm_new = self._pm_update(pm, llr, u)
                    B[l, n] = u
                    u_hat[l] = u
                    _update_bits(B, l, n, N)
                    new_paths.append({
                        'L': L, 'B': B, 'pm': pm_new,
                        'u_hat': u_hat.copy(),
                    })
                else:
                    for u in (0, 1):
                        Lc = L.copy()
                        Bc = B.copy()
                        uhc = u_hat.copy()
                        pm_new = self._pm_update(pm, llr, u)
                        Bc[l, n] = u
                        uhc[l] = u
                        _update_bits(Bc, l, n, N)
                        new_paths.append({
                            'L': Lc, 'B': Bc, 'pm': pm_new, 'u_hat': uhc,
                        })

            new_paths.sort(key=lambda p: p['pm'])
            paths = new_paths[:self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if self._crc_valid(p['u_hat'])]
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p['pm'])
        return best['u_hat'], best['pm']

    def _crc_valid(self, u_hat):
        info_bits = u_hat[self.info_indices]
        if len(info_bits) < self.crc_length:
            return False
        return crc_check(info_bits, self.crc_length)
