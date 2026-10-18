"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import os
import sys

import numpy as np

from decoder_sc import sc_decode_recursive

_REF_DIR = os.path.join(os.path.dirname(__file__), "ref_polar")
if _REF_DIR not in sys.path:
    sys.path.insert(0, _REF_DIR)

import decoder as _ref_decoder  # noqa: E402

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class SCLDecoder:
    """SCL 译码：L=1 走自研 SC；L>1 使用 ref_polar 步进 SCL（与译码域 LLR 对齐）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.L_size = max(1, int(list_size))
        self.crc_length = int(crc_length)
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        from channel import channel_llr_to_decode_domain

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_dec = channel_llr_to_decode_domain(llr_ch)

        if self.L_size == 1:
            u = sc_decode_recursive(llr_dec, self.frozen_bits)
            return u, 0.0

        information_pos = list(self.info_indices)
        frozen_bit = 0
        crc_n = self.crc_length if self.crc_length > 0 else 8
        u_hat = _ref_decoder.scl_decoder(
            llr_dec, information_pos, frozen_bit, [self.L_size, "hf"], crc_n
        )
        u_hat = np.asarray(u_hat, dtype=int)
        return u_hat, 0.0
