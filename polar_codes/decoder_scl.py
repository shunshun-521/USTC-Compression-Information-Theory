"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation
from decoder_sc import _compute_llr


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_poly_bits(crc_length):
    if crc_length == 8:
        poly = CRC8_POLY
    elif crc_length == 16:
        poly = CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    bits = []
    for shift in range(crc_length, -1, -1):
        bits.append((poly >> shift) & 1)
    return np.array(bits[1:], dtype=int)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_poly_bits(crc_length)
    reg = np.zeros(crc_length, dtype=int)
    for bit in info_bits:
        feedback = bit ^ reg[0]
        reg[:-1] = reg[1:]
        reg[-1] = 0
        if feedback:
            reg ^= poly
    return np.concatenate([info_bits, reg])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    recomputed = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(recomputed[-crc_length:], bits[-crc_length:])


def _path_metric_penalty(llr_val, u_bit):
    preferred = 0 if llr_val >= 0 else 1
    return 0.0 if u_bit == preferred else abs(llr_val)


class SCLDecoder:
    """SCL 译码器（Lazy Copy）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)

    def _new_state(self, llr_internal):
        llrs = np.full((self.n + 1, self.N), -np.inf, dtype=np.float64)
        bits = np.full((self.n + 1, self.N), -1, dtype=np.int64)
        llrs[self.n, :] = llr_internal
        return llrs, bits, 0.0, np.zeros(self.N, dtype=int)

    def decode(self, llr_ch):
        """返回 (u_hat, pm)。"""
        llr_internal = np.asarray(llr_ch, dtype=np.float64)[self.br]
        L = self.list_size
        N = self.N

        states = [self._new_state(llr_internal)]

        for phi in range(N):
            expanded = []
            for llrs, bits, pm, u_hat in states:
                llr_phi = _compute_llr(0, phi, llrs, bits)

                if self.frozen_bits[phi]:
                    pen = _path_metric_penalty(llr_phi, 0)
                    bits[0, phi] = 0
                    u_hat[phi] = 0
                    expanded.append((llrs, bits, pm + pen, u_hat.copy()))
                else:
                    for u_bit in (0, 1):
                        llrs_copy = llrs.copy()
                        bits_copy = bits.copy()
                        u_copy = u_hat.copy()
                        pen = _path_metric_penalty(llr_phi, u_bit)
                        bits_copy[0, phi] = u_bit
                        u_copy[phi] = u_bit
                        expanded.append((llrs_copy, bits_copy, pm + pen, u_copy))

            expanded.sort(key=lambda item: item[2])
            states = expanded[:L]

        if self.crc_length > 0:
            info_positions = np.where(~self.frozen_bits)[0]
            valid = [
                st for st in states
                if crc_check(st[3][info_positions], self.crc_length)
            ]
            if valid:
                best = min(valid, key=lambda st: st[2])
                return best[3].copy(), best[2]

        best = min(states, key=lambda st: st[2])
        return best[3].copy(), best[2]
