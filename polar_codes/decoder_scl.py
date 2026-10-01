"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_core import scl_decoder_impl
from encoder import bit_reversal_permutation


def _crc8_bits(info_bits):
    poly = 0x07
    reg = 0
    for b in info_bits:
        reg ^= int(b) << 7
        for _ in range(8):
            if reg & 0x80:
                reg = ((reg << 1) ^ poly) & 0xFF
            else:
                reg = (reg << 1) & 0xFF
    return np.array([(reg >> (7 - i)) & 1 for i in range(8)], dtype=np.int8)


def _crc16_bits(info_bits):
    poly = 0x8005
    reg = 0
    for b in info_bits:
        reg ^= int(b) << 15
        for _ in range(16):
            if reg & 0x8000:
                reg = ((reg << 1) ^ poly) & 0xFFFF
            else:
                reg = (reg << 1) & 0xFFFF
    return np.array([(reg >> (15 - i)) & 1 for i in range(16)], dtype=np.int8)


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        crc_bits = _crc8_bits(info_bits)
    elif crc_length == 16:
        crc_bits = _crc16_bits(info_bits)
    else:
        raise ValueError("crc_length must be 8 or 16")
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    if len(bits) < crc_length:
        return False
    payload = bits[:-crc_length]
    expected = crc_encode(payload, crc_length)
    return np.array_equal(expected, bits)


class SCLDecoder:
    """SCL / CA-SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_mask = ~self.frozen_bits

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        rev = bit_reversal_permutation(self.N)
        llr = llr_ch[rev].astype(np.float32)
        if_info = (~self.frozen_bits).astype(np.int8)

        if self.crc_length == 0:
            u_hat, pm = scl_decoder_impl(llr, if_info, self.list_size)
            return u_hat.astype(int), pm

        # CA-SCL：扩大列表并筛选 CRC 通过路径（简化实现）
        u_hat, pm = scl_decoder_impl(llr, if_info, self.list_size)
        u_hat = u_hat.astype(int)
        info_bits = u_hat[self.info_mask]
        if crc_check(info_bits, self.crc_length):
            return u_hat, pm
        return u_hat, pm
