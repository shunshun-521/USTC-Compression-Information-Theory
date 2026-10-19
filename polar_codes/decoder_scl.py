"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math
from decoder_sc import f_operation, sc_decode_recursive


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        if reg & top:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        if reg & top:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg == 0


def _pm_add(llr, u_bit):
    hard = 0 if llr >= 0 else 1
    return 0.0 if u_bit == hard else abs(llr)


class SCLDecoder:
    """SCL 译码器（树结构与 SC 一致，路径度量 + 剪枝）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = None if info_indices is None else np.asarray(info_indices, dtype=int)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1:
            return sc_decode_recursive(llr_ch, self.frozen_bits), 0.0

        n = self.n
        frozen = self.frozen_bits
        L = self.list_size

        def decode_paths(paths_in, depth, bit_position):
            if depth == n - 1:
                out = []
                for pm, llrs, bits in paths_in:
                    left_bp = 2 * bit_position
                    Lu = float(f_operation(llrs[0], llrs[1]))
                    for u_bit in ([0] if frozen[left_bp] else [0, 1]):
                        pm2 = pm + _pm_add(Lu, u_bit)
                        bits2 = bits.copy()
                        bits2[left_bp] = u_bit
                        u_sign = 1.0 if u_bit == 0 else -1.0
                        Lv = float(u_sign * llrs[0] + llrs[1])
                        right_bp = 2 * bit_position + 1
                        for v_bit in ([0] if frozen[right_bp] else [0, 1]):
                            pm3 = pm2 + _pm_add(Lv, v_bit)
                            bits3 = bits2.copy()
                            bits3[right_bp] = v_bit
                            v_sign = 1.0 if v_bit == 0 else -1.0
                            signs = np.array([u_sign * v_sign, v_sign], dtype=np.float64)
                            out.append((pm3, llrs, bits3, signs))
                out.sort(key=lambda x: x[0])
                return out[:L]

            out = []
            half_index = 2 ** (n - depth - 1)
            for pm, llrs, bits in paths_in:
                Lu = f_operation(llrs[:half_index], llrs[half_index:])
                left_res = decode_paths([(pm, Lu, bits)], depth + 1, bit_position * 2)
                for pm_l, _, bits_l, u_hat in left_res:
                    Lv = u_hat * llrs[:half_index] + llrs[half_index:]
                    right_res = decode_paths([(pm_l, Lv, bits_l)], depth + 1, bit_position * 2 + 1)
                    for pm_r, _, bits_r, v_hat in right_res:
                        signs = np.concatenate((u_hat * v_hat, v_hat))
                        out.append((pm_r, llrs, bits_r, signs))
            out.sort(key=lambda x: x[0])
            return out[:L]

        paths = decode_paths([(0.0, llr_ch.copy(), np.zeros(self.N, dtype=int))], 0, 0)
        best = paths[0]
        bits = best[2]

        if self.crc_length > 0:
            check_bits = (
                bits[self.info_indices] if self.info_indices is not None else bits
            )
            if not crc_check(check_bits, self.crc_length):
                for p in paths:
                    cb = (
                        p[2][self.info_indices]
                        if self.info_indices is not None
                        else p[2]
                    )
                    if crc_check(cb, self.crc_length):
                        best = p
                        break
        return best[2].copy(), best[0]
