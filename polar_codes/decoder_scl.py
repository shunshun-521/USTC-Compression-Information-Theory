"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import _scl_decode_core


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def _crc_register_update(reg, bit, poly, crc_length):
    mask = (1 << crc_length) - 1
    fb = ((reg >> (crc_length - 1)) ^ int(bit)) & 1
    reg = (reg << 1) & mask
    if fb:
        reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for b in info_bits:
        reg = _crc_register_update(reg, b, poly, crc_length)
    crc_bits = np.array(
        [(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for b in bits:
        reg = _crc_register_update(reg, b, poly, crc_length)
    return reg == 0


class SCLDecoder:
    """SCL 译码器（log-domain 路径度量）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        paths = _scl_decode_core(
            llr_ch, self.frozen_bits, list_size=self.list_size
        )
        if self.crc_length > 0:
            for metric, u_hat, _ in paths:
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    return u_hat.astype(int), metric
        best = paths[0]
        return best[1].astype(int), best[0]
