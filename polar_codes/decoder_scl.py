"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import sc_decode
import _ref_decoder as ref_decoder


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07; CRC-16: 0x8005
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = 0x07 if crc_length == 8 else 0x8005

    if crc_length == 8:
        reg = 0
        for bit in info_bits:
            reg ^= (int(bit) << 7)
            for _ in range(8):
                if reg & 0x80:
                    reg = ((reg << 1) ^ poly) & 0xFF
                else:
                    reg = (reg << 1) & 0xFF
        crc_bits = np.array([(reg >> (7 - i)) & 1 for i in range(8)], dtype=int)
    else:
        reg = 0
        for bit in info_bits:
            reg ^= (int(bit) << 15)
            for _ in range(16):
                if reg & 0x8000:
                    reg = ((reg << 1) ^ poly) & 0xFFFF
                else:
                    reg = (reg << 1) & 0xFFFF
        crc_bits = np.array([(reg >> (15 - i)) & 1 for i in range(16)], dtype=int)

    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC，返回 True/False"""
    bits = np.asarray(bits, dtype=int)
    info = bits[:-crc_length]
    expected = crc_encode(info, crc_length)
    return np.array_equal(bits, expected)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.br = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        if self.list_size == 1:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        y_llr = llr_ch[self.br]
        decode_para = [self.list_size, 'hf']

        if self.crc_length > 0:
            u_hat = ref_decoder.scl_decoder(
                y_llr, self.info_indices, 0, decode_para, self.crc_length
            )
        else:
            u_hat = ref_decoder.scl_decoder(
                y_llr, self.info_indices, 0, decode_para, 0
            )

        u_hat = np.array([0 if u_hat[i] == 0 else 1 for i in range(u_hat.size)])
        return u_hat.astype(int), 0.0
