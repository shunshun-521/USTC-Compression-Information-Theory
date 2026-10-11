"""
极化码 SCL（串行抵消列表）译码器，支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import sc_decode
from sc_backend import scl_decode_core


def _crc8_bits(data_bits):
    poly = 0x07
    reg = 0
    for b in data_bits:
        reg ^= int(b) << 7
        for _ in range(8):
            reg = ((reg << 1) ^ poly) & 0xFF if (reg & 0x80) else (reg << 1) & 0xFF
    return [(reg >> (7 - i)) & 1 for i in range(8)]


def _crc16_bits(data_bits):
    poly = 0x8005
    reg = 0
    for b in data_bits:
        reg ^= int(b) << 15
        for _ in range(16):
            reg = ((reg << 1) ^ poly) & 0xFFFF if (reg & 0x8000) else (reg << 1) & 0xFFFF
    return [(reg >> (15 - i)) & 1 for i in range(16)]


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    if crc_length == 8:
        crc = _crc8_bits(info_bits)
    elif crc_length == 16:
        crc = _crc16_bits(info_bits)
    else:
        raise ValueError("crc_length must be 8 or 16")
    return np.concatenate([info_bits, np.array(crc, dtype=int)])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int).ravel()
    if len(bits) < crc_length:
        return False
    return np.array_equal(bits, crc_encode(bits[:-crc_length], crc_length))


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.info_idx = np.where(~self.frozen_bits)[0]
        self.list_size = max(1, int(list_size))
        self.crc_length = int(crc_length)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size <= 1:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        u_hat, pm = scl_decode_core(
            llr_ch,
            self.info_idx.tolist(),
            self.list_size,
            frozen_bit=0,
        )

        if self.crc_length > 0:
            info_only = u_hat[self.info_idx]
            if crc_check(info_only, self.crc_length):
                return u_hat, pm
        return u_hat, pm
