"""
极化码 SCL（串行抵消列表）译码器，含 CRC 辅助 CA-SCL
"""
import numpy as np

import ref_crc as CRC
import ref_scl_bp as ref_dec
from decoder_sc import sc_decode


def crc_encode(info_bits, crc_length=8):
    bits = list(np.asarray(info_bits, dtype=int).tolist())
    coder = CRC.CRC(bits, crc_length)
    return np.array(coder.code, dtype=np.int8)


def crc_check(bits, crc_length=8):
    bits = list(np.asarray(bits, dtype=int).tolist())
    return bool(CRC.CRC(bits, crc_length).detection())


class SCLDecoder:
    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.information_pos = np.where(~self.frozen_bits)[0].tolist()
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_bit = 0

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size <= 1 and self.crc_length == 0:
            u = sc_decode(llr_ch, self.frozen_bits)
            return u, 0.0

        decode_para = [self.list_size, "exact"]
        u_d = ref_dec.scl_decoder(
            llr_ch,
            self.information_pos,
            self.frozen_bit,
            decode_para,
            self.crc_length,
        )
        u_d = np.array(u_d, dtype=np.int8)
        u_d[self.frozen_bits] = 0
        return u_d, 0.0
