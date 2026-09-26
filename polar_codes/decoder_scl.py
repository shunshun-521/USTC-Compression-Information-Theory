"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import sc_decode, bit_llr_at_phi


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_update(reg, bit, poly, crc_length):
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    reg ^= int(bit) << (crc_length - 1)
    if reg & top:
        reg = ((reg << 1) ^ poly) & mask
    else:
        reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    reg = 0
    for b in info_bits:
        reg = _crc_update(reg, b, poly, crc_length)
    crc_bits = [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)]
    return np.concatenate([info_bits, np.array(crc_bits, dtype=np.int8)])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    reg = 0
    for b in bits:
        reg = _crc_update(reg, b, poly, crc_length)
    return reg == 0


def _pm_add(pm, llr_val, u_bit):
    hard = 0 if llr_val >= 0 else 1
    if u_bit != hard:
        pm += abs(llr_val)
    return pm


class SCLDecoder:
    """SCL 译码器（逐比特路径扩展）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=np.float64)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits < 0.5)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1:
            u = sc_decode(llr_ch, self.frozen_bits)
            return u, 0.0
        paths = [(0.0, np.zeros(self.N, dtype=int))]

        for phi in range(self.N):
            new_paths = []
            for pm, u in paths:
                llr_phi = bit_llr_at_phi(llr_ch, self.frozen_bits, u, phi)
                if self.frozen_bits[phi] >= 0.5:
                    u[phi] = 0
                    new_paths.append((_pm_add(pm, llr_phi, 0), u.copy()))
                else:
                    for bit in (0, 1):
                        u2 = u.copy()
                        u2[phi] = bit
                        new_paths.append((_pm_add(pm, llr_phi, bit), u2))
            new_paths.sort(key=lambda x: x[0])
            paths = new_paths[: self.list_size]

        paths.sort(key=lambda x: x[0])
        if self.crc_length > 0:
            valid = []
            for pm, u in paths:
                info = u[self.info_indices]
                if crc_check(info, self.crc_length):
                    valid.append((pm, u))
            if valid:
                paths = valid

        best = paths[0][1]
        return best.astype(int), paths[0][0]
