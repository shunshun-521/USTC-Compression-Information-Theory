"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    f_operation,
    _b_check,
    _s_updater,
    _compute_llr,
    _to_info_mask,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    r=8: CRC-8 (0x07), r=16: CRC-16 (0x8005)
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY

    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits)
        self.info_mask = _to_info_mask(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length

    def _init_path(self, llr):
        llrs = -np.inf * np.ones((self.n + 1, self.N), dtype=np.float64)
        llrs[-1, :] = llr
        bits = -np.ones((self.n + 1, self.N), dtype=np.int8)
        return llrs, bits, 0.0, np.zeros(self.N, dtype=int)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, pm
        """
        from encoder import bit_reversal_permutation

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        br = bit_reversal_permutation(N)
        llr = llr_ch[br].copy()

        paths = [self._init_path(llr)]

        for phi in range(N):
            candidates = []

            for llrs, bits, pm, u_hat in paths:
                if self.info_mask[phi] == 0:
                    llr_val = _compute_llr(0, phi, llrs, bits)
                    new_llrs = llrs.copy()
                    new_bits = bits.copy()
                    new_u = u_hat.copy()
                    penalty = abs(llr_val) if llr_val < 0 else 0.0
                    new_pm = pm + penalty
                    new_u[phi] = 0
                    new_bits[0, phi] = 0
                    new_llrs[0, phi] = np.inf
                    candidates.append((new_llrs, new_bits, new_pm, new_u))
                else:
                    llr_val = _compute_llr(0, phi, llrs, bits)
                    for bit in (0, 1):
                        new_llrs = llrs.copy()
                        new_bits = bits.copy()
                        new_u = u_hat.copy()
                        hard = 1 if llr_val < 0 else 0
                        penalty = 0.0 if bit == hard else abs(llr_val)
                        new_pm = pm + penalty
                        new_u[phi] = bit
                        new_bits[0, phi] = bit
                        candidates.append((new_llrs, new_bits, new_pm, new_u))

            candidates.sort(key=lambda item: item[2])
            paths = candidates[:self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p[3], self.crc_length)]
            best = min(valid, key=lambda p: p[2]) if valid else min(paths, key=lambda p: p[2])
        else:
            best = min(paths, key=lambda p: p[2])

        return best[3].copy(), best[2]
