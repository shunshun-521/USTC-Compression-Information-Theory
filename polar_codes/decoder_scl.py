"""
极化码 SCL（串行抵消列表）译码器，含 CRC 辅助 CA-SCL
"""
import numpy as np
from crc_helper import CRC
from _codec_core import scl_decoder as _scl_decoder


def _info_indices(frozen_bits):
    return np.where(np.asarray(frozen_bits) == 0)[0]


def crc_encode(info_bits, crc_length=8):
    """CRC 编码，返回信息位 + 校验位"""
    info_bits = list(np.asarray(info_bits, dtype=int).ravel())
    return np.array(CRC(info_bits, crc_length).code, dtype=int)


def crc_check(bits, crc_length=8):
    """校验信息位 + CRC"""
    bits = list(np.asarray(bits, dtype=int).ravel())
    if len(bits) < crc_length:
        return False
    info = bits[:-crc_length]
    expected = CRC(info, crc_length).code
    return bits == expected


class SCLDecoder:
    """SCL / CA-SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.info_idx = _info_indices(self.frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        decode_para = [self.list_size, "hf"]
        u_hat = _scl_decoder(
            llr_ch, self.info_idx, 0, decode_para, self.crc_length
        )
        u_hat = np.asarray(u_hat, dtype=int)
        return u_hat, 0.0
