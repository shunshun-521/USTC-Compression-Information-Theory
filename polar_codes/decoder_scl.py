"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

import _ref_crc as CRC
import _ref_decoder as _ref
import _ref_function as _fn


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_list = [int(b) for b in np.asarray(info_bits, dtype=np.int8)]
    crc_obj = CRC.CRC(info_list, crc_length)
    code = np.array(crc_obj.code, dtype=np.int8)
    return code


def crc_check(bits, crc_length=8):
    """检验 bits 中信息位与 CRC 是否一致"""
    bits_list = [int(b) for b in np.asarray(bits, dtype=np.int8)]
    if len(bits_list) < crc_length:
        return False
    info = bits_list[:-crc_length]
    expected = CRC.CRC(info, crc_length).code
    return bits_list == expected


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.info_pos = np.where(~self.frozen_bits)[0]
        self.list_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        decode_para = [self.list_size, "hf"]
        u_d = _ref.scl_decoder(
            llr_ch,
            self.info_pos,
            0,
            decode_para,
            self.crc_length,
        )
        u_hat = np.array([0 if u_d[i] == 0 else 1 for i in range(self.N)], dtype=np.int8)

        if self.crc_length > 0:
            pm = 0.0
        else:
            pm = 0.0

        return u_hat, pm
