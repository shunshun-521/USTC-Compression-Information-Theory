"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import sc_decode_recursive, sc_leaf_llr


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_len):
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_len - 1)
        for _ in range(8):
            if reg & (1 << (crc_len - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_len) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_len) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    @staticmethod
    def _pm_penalty(llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        if self.list_size == 1:
            u = sc_decode_recursive(llr_ch, self.frozen_bits)
            return u, 0.0

        paths = [(0.0, np.zeros(self.N, dtype=int))]
        for phi in range(self.N):
            new_paths = []
            for pm, u_hat in paths:
                leaf = sc_leaf_llr(llr_ch, self.frozen_bits, u_hat, phi)
                if self.frozen_bits[phi]:
                    pm2 = pm + self._pm_penalty(leaf, 0)
                    u2 = u_hat.copy()
                    u2[phi] = 0
                    new_paths.append((pm2, u2))
                else:
                    for bit in (0, 1):
                        pm2 = pm + self._pm_penalty(leaf, bit)
                        u2 = u_hat.copy()
                        u2[phi] = bit
                        new_paths.append((pm2, u2))
            new_paths.sort(key=lambda x: x[0])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for pm, u_hat in paths:
                info = u_hat[self.info_indices]
                if crc_check(info, self.crc_length):
                    valid.append((pm, u_hat))
            if valid:
                paths = valid

        pm, u_hat = min(paths, key=lambda x: x[0])
        return u_hat, pm
