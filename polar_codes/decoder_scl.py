"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversed, bit_reversal_permutation
from decoder_sc import (
    f_operation, g_operation, _active_llr_level, _active_bit_level,
    _update_llrs, _update_bits,
)


CRC_POLYNOMIALS = {8: 0x07, 16: 0x8005}


def _crc_bits(bits, poly, crc_length):
    crc = 0
    mask = (1 << crc_length) - 1
    for bit in bits:
        crc ^= int(bit) << (crc_length - 1)
        if crc & (1 << (crc_length - 1)):
            crc = ((crc << 1) ^ poly) & mask
        else:
            crc = (crc << 1) & mask
    return crc


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC_POLYNOMIALS[crc_length]
    remainder = _crc_bits(list(info_bits) + [0] * crc_length, poly, crc_length)
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
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits, expected)


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.br = bit_reversal_permutation(N)

    def _path_llr(self, path, l, n):
        return path['L'][l, n]

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)。"""
        N = self.N
        n = self.n
        llr = np.asarray(llr_ch, dtype=np.float64)[self.br]

        def new_path():
            return {
                'L': np.zeros((N, n + 1), dtype=np.float64),
                'B': np.zeros((N, n + 1), dtype=np.int32),
                'pm': 0.0,
            }

        paths = [new_path()]
        paths[0]['L'][:, 0] = llr

        for phi in range(N):
            l = bit_reversed(phi, n)
            candidates = []

            for path in paths:
                _update_llrs(path['L'], path['B'], l, n)
                llr_val = path['L'][l, n]

                if l in self.frozen_set:
                    new_path = {'L': path['L'], 'B': path['B'].copy(), 'pm': path['pm']}
                    penalty = 0.0 if llr_val >= 0 else abs(llr_val)
                    new_path['pm'] += penalty
                    new_path['B'][l, n] = 0
                    _update_bits(new_path['B'], l, n)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = {
                            'L': path['L'],
                            'B': path['B'].copy(),
                            'pm': path['pm'],
                        }
                        llr_bit = 0 if llr_val >= 0 else 1
                        penalty = 0.0 if bit == llr_bit else abs(llr_val)
                        new_path['pm'] += penalty
                        new_path['B'][l, n] = bit
                        _update_bits(new_path['B'], l, n)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p['pm'])
            # Lazy copy: duplicate L only when path survives pruning
            pruned = candidates[:self.list_size]
            for p in pruned:
                p['L'] = p['L'].copy()
            paths = pruned

        if self.crc_length > 0:
            valid = []
            for path in paths:
                info_bits = path['B'][:, n][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            best = min(valid if valid else paths, key=lambda p: p['pm'])
        else:
            best = min(paths, key=lambda p: p['pm'])

        return best['B'][:, n].copy(), best['pm']
