"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    _sc_decode_core, _bit_reversed, _active_llr_level, _active_bit_level,
    _upper_llr, _lower_llr,
)


CRC_POLYS = {8: 0x07, 16: 0x8005}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    poly = CRC_POLYS[crc_length]
    info_bits = np.asarray(info_bits, dtype=int)
    reg = 0
    for bit in info_bits:
        reg ^= (int(bit) << (crc_length - 1))
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    expected_crc = crc_encode(bits[:-crc_length], crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected_crc)


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)

    def _pm_penalty(self, llr_val, u):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u == hard else abs(llr_val)

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)。"""
        llr = llr_ch[self.br].astype(np.float64)
        N, n = self.N, self.n

        paths = [{
            'L': np.full((N, n + 1), np.nan, dtype=np.float64),
            'B': np.full((N, n + 1), np.nan),
            'pm': 0.0,
        }]
        paths[0]['L'][:, 0] = llr

        for i in range(N):
            l = _bit_reversed(i, n)
            new_paths = []

            for path in paths:
                L, B, pm = path['L'], path['B'], path['pm']

                for s in range(n - _active_llr_level(l, n), n):
                    bs = 2 ** (s + 1)
                    brs = bs // 2
                    for j in range(l, N, bs):
                        if j % bs < brs:
                            L[j, s + 1] = _upper_llr(L[j, s], L[j + brs, s])
                        else:
                            L[j, s + 1] = _lower_llr(
                                L[j, s], L[j - brs, s], int(B[j - brs, s + 1])
                            )

                llr_val = L[l, n]

                if l in self.frozen_set:
                    pen = self._pm_penalty(llr_val, 0)
                    np_path = {
                        'L': L.copy(),
                        'B': B.copy(),
                        'pm': pm + pen,
                    }
                    np_path['B'][l, n] = 0
                    if l >= N / 2:
                        self._update_bits(np_path['B'], l, n)
                    new_paths.append(np_path)
                else:
                    for u in (0, 1):
                        pen = self._pm_penalty(llr_val, u)
                        np_path = {
                            'L': L.copy(),
                            'B': B.copy(),
                            'pm': pm + pen,
                        }
                        np_path['B'][l, n] = u
                        if l >= N / 2:
                            self._update_bits(np_path['B'], l, n)
                        new_paths.append(np_path)

            new_paths.sort(key=lambda p: p['pm'])
            paths = new_paths[:self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                u_hat = p['B'][:, n].astype(int)
                info_bits = u_hat[~self.frozen_bits]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            best = min(valid if valid else paths, key=lambda p: p['pm'])
        else:
            best = min(paths, key=lambda p: p['pm'])

        return best['B'][:, n].astype(int), best['pm']

    @staticmethod
    def _update_bits(B, l, n):
        if l < len(B) / 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            bs = 2 ** s
            brs = bs // 2
            for j in range(l, -1, -bs):
                if j % bs >= brs:
                    B[j - brs, s - 1] = int(B[j, s]) ^ int(B[j - brs, s])
                    B[j, s - 1] = B[j, s]
