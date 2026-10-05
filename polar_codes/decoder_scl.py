"""
极化码 SCL（串行抵消列表）译码器，支持 CRC 辅助 CA-SCL
"""
import numpy as np
import _shipeng_decoder as _sd
import _shipeng_crc as _crc_mod


def crc_encode(info_bits, crc_length=8):
    """CRC-8 (0x07) 或 CRC-16 (0x8005)"""
    bits = list(np.asarray(info_bits, dtype=int).flatten())
    coder = _crc_mod.CRC(bits, crc_length)
    full = coder.code
    return np.array(full, dtype=int)


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    seq = list(np.asarray(bits, dtype=int).flatten())
    return _crc_mod.CRC(seq, crc_length).detection() == 1


class SCLDecoder:
    """SCL 译码器（封装列表 SC + CRC 选择）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.info_indices = list(np.where(self.frozen_bits == 0)[0])
        self.list_size = int(list_size)
        self.crc_length = int(crc_length)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1 and self.crc_length == 0:
            from decoder_sc import sc_decode

            return sc_decode(llr_ch, self.frozen_bits), 0.0
        decode_para = [self.list_size, "hf"]
        u_hat = _sd.scl_decoder(
            llr_ch,
            self.info_indices,
            0,
            decode_para,
            self.crc_length,
        )
        u_hat = np.asarray(u_hat, dtype=int)
        pm = 0.0
        return u_hat, pm
