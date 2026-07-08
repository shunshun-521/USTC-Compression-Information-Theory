"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    f_operation,
    g_operation,
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _update_bits,
    _update_llrs,
)
from utils import crc_check, crc_encode


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.L_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        """
        主译码函数。
        返回 (u_hat, pm)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        Lsz = self.L_size

        paths = [{
            'L': np.zeros((N, n + 1), dtype=np.float64),
            'B': np.zeros((N, n + 1), dtype=np.int8),
            'pm': 0.0,
            'u_hat': np.zeros(N, dtype=int),
        }]
        paths[0]['L'][:, 0] = llr_ch

        for i in range(N):
            l = _bit_reversed(i, n)
            candidates = []

            for pidx, path in enumerate(paths):
                L, B = path['L'], path['B']
                _update_llrs(L, B, l, n)
                cur_llr = L[l, n]

                if l in self.frozen_set:
                    penalty = 0.0 if cur_llr >= 0 else abs(cur_llr)
                    new_path = self._lazy_copy(path)
                    new_path['pm'] += penalty
                    new_path['B'][l, n] = 0
                    new_path['u_hat'][l] = 0
                    _update_bits(new_path['B'], l, n)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        penalty = 0.0 if (bit == 0 and cur_llr >= 0) or (bit == 1 and cur_llr < 0) else abs(cur_llr)
                        new_path = self._lazy_copy(path)
                        new_path['pm'] += penalty
                        new_path['B'][l, n] = bit
                        new_path['u_hat'][l] = bit
                        _update_bits(new_path['B'], l, n)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p['pm'])
            paths = candidates[:Lsz]

        if self.crc_length > 0:
            valid = [p for p in paths if self._crc_pass(p['u_hat'])]
            if valid:
                best = min(valid, key=lambda p: p['pm'])
            else:
                best = min(paths, key=lambda p: p['pm'])
        else:
            best = min(paths, key=lambda p: p['pm'])

        return best['u_hat'], best['pm']

    def _lazy_copy(self, path):
        """浅拷贝路径状态（数组引用共享，写时复制由新赋值保证）"""
        return {
            'L': path['L'].copy(),
            'B': path['B'].copy(),
            'pm': path['pm'],
            'u_hat': path['u_hat'].copy(),
        }

    def _crc_pass(self, u_hat):
        info_idx = np.where(~self.frozen_bits)[0]
        info_bits = u_hat[info_idx]
        return crc_check(info_bits, self.crc_length)


# 导出 CRC 函数以保持模块接口一致
__all__ = ['SCLDecoder', 'crc_encode', 'crc_check']
