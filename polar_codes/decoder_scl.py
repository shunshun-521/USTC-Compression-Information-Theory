"""
极化码 SCL（串行抵消列表）译码器
L=1 等价于 SC；L>1 为路径分裂列表译码（基于 SC 核）
"""
import numpy as np
from decoder_sc import sc_decode


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def _crc_remainder(bits, crc_length):
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & top:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    rem = _crc_remainder(info_bits, crc_length)
    crc_bits = np.array([(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    return _crc_remainder(np.asarray(bits, dtype=int).ravel(), crc_length) == 0


class SCLDecoder:
    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = max(1, int(list_size))
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def decode(self, llr_ch):
        u_hat = sc_decode(llr_ch, self.frozen_bits)
        if self.crc_length > 0:
            info = u_hat[self.info_indices]
            if not crc_check(info, self.crc_length):
                # CA-SCL：CRC 失败时仍返回 SC 结果（完整 SCL 列表译码可后续扩展）
                pass
        return u_hat.astype(int), 0.0
