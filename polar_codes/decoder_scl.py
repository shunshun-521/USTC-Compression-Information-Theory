"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import _update_llrs, _update_bits


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY

    reg = 0
    mask = (1 << crc_length) - 1
    for bit in info_bits:
        msb = (reg >> (crc_length - 1)) & 1
        reg = ((reg << 1) | int(bit)) & mask
        if msb ^ int(bit):
            for i in range(crc_length):
                if (poly >> i) & 1:
                    reg ^= 1 << i

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """
    检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。
    返回 True/False。
    """
    bits = np.asarray(bits, dtype=np.int8)
    info = bits[:-crc_length]
    expected = crc_encode(info, crc_length=crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


class SCLDecoder:
    """
    SCL 译码器。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = None if info_indices is None else np.asarray(info_indices, dtype=int)
        self.br = bit_reversal_permutation(N)

    @staticmethod
    def _pm_update(pm, llr, bit):
        hard = 0 if llr >= 0 else 1
        if bit != hard:
            return pm + abs(llr)
        return pm

    def _crc_valid(self, u_hat):
        if self.crc_length <= 0:
            return True
        if self.info_indices is None:
            return crc_check(u_hat, self.crc_length)
        info_bits = u_hat[self.info_indices]
        return crc_check(info_bits, self.crc_length)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：
            u_hat: 长度 N 的估计源序列（最优路径）
            pm: 最优路径的度量值
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        paths = [{
            'pm': 0.0,
            'L': np.full((N, n + 1), np.nan, dtype=np.float64),
            'B': np.full((N, n + 1), np.nan),
        }]
        paths[0]['L'][:, 0] = llr_ch

        for phi in range(N):
            l = self.br[phi]
            candidates = []

            for path in paths:
                _update_llrs(path['L'], path['B'], l, n)
                llr = path['L'][l, n]

                if self.frozen_bits[l]:
                    new_path = {
                        'pm': self._pm_update(path['pm'], llr, 0),
                        'L': path['L'].copy(),
                        'B': path['B'].copy(),
                    }
                    new_path['B'][l, n] = 0
                    _update_bits(new_path['B'], l, n, N)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = {
                            'pm': self._pm_update(path['pm'], llr, bit),
                            'L': path['L'].copy(),
                            'B': path['B'].copy(),
                        }
                        new_path['B'][l, n] = bit
                        _update_bits(new_path['B'], l, n, N)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p['pm'])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if self._crc_valid(p['B'][:, n].astype(int))]
            best = min(valid, key=lambda p: p['pm']) if valid else paths[0]
        else:
            best = paths[0]

        u_hat = best['B'][:, n].astype(int)
        return u_hat, best['pm']
