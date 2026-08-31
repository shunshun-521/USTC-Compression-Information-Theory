"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from polar_ops import f_min_sum, reorder_channel_llr


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    mask = (1 << crc_length) - 1
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class SCLDecoder:
    """SCL 译码器（基于因子图逐比特扩展）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.LARGE = 1e6

    def _bit_llr(self, llr_ch, u_partial, phi):
        """计算比特 phi 处的 LLR，已知 u_partial[0:phi]。"""
        N, n = self.N, self.n
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        for i in range(phi):
            if self.frozen_bits[i]:
                R[i, 0] = self.LARGE
            else:
                R[i, 0] = self.LARGE if u_partial[i] == 0 else -self.LARGE

        for j in range(n, 0, -1):
            s = 1 << (j - 1)
            for i in range(0, N, 2 * s):
                for k in range(s):
                    idx = i + k
                    L[idx, j - 1] = f_min_sum(
                        R[idx, j - 1] + L[idx + s, j], L[idx, j], 1.0
                    )
                    L[idx + s, j - 1] = (
                        f_min_sum(R[idx, j - 1], L[idx, j], 1.0) + L[idx + s, j]
                    )

        return L[phi, 0] + R[phi, 0]

    def decode(self, llr_ch):
        """主译码函数。"""
        llr_ch = reorder_channel_llr(llr_ch)
        paths = [(0.0, np.zeros(self.N, dtype=int))]

        for phi in range(self.N):
            candidates = []
            for pm, bits in paths:
                llr = self._bit_llr(llr_ch, bits, phi)
                if self.frozen_bits[phi]:
                    new_bits = bits.copy()
                    new_bits[phi] = 0
                    penalty = abs(llr) if llr < 0 else 0.0
                    candidates.append((pm + penalty, new_bits))
                else:
                    for u in (0, 1):
                        new_bits = bits.copy()
                        new_bits[phi] = u
                        consistent = (u == 0 and llr >= 0) or (u == 1 and llr < 0)
                        penalty = 0.0 if consistent else abs(llr)
                        candidates.append((pm + penalty, new_bits))

            candidates.sort(key=lambda x: x[0])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [
                (pm, bits)
                for pm, bits in paths
                if crc_check(bits[self.info_indices], self.crc_length)
            ]
            if valid:
                paths = valid

        best_pm, best_bits = min(paths, key=lambda x: x[0])
        return best_bits.copy(), best_pm
