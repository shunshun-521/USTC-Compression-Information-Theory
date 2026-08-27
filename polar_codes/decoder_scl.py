"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from polar_core import SCLDecoder as _SCLDecoder, SCLCRCDecoder as _SCLCRCDecoder


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        poly = np.array([1, 0, 0, 0, 0, 0, 1, 1, 1], dtype=np.int8)
    elif crc_length == 16:
        poly = np.array(
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=np.int8
        )
    else:
        raise ValueError("crc_length must be 8 or 16")
    r = len(poly) - 1
    msg = np.concatenate([info_bits, np.zeros(r, dtype=np.int8)])
    for i in range(len(info_bits)):
        if msg[i] == 1:
            msg[i:i + len(poly)] ^= poly
    return np.concatenate([info_bits, msg[len(info_bits):]])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=np.int8)
    if crc_length == 8:
        poly = np.array([1, 0, 0, 0, 0, 0, 1, 1, 1], dtype=np.int8)
    else:
        poly = np.array(
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=np.int8
        )
    r = len(poly) - 1
    msg = bits.copy()
    for i in range(len(bits) - r):
        if msg[i] == 1:
            msg[i:i + len(poly)] ^= poly
    return np.all(msg[-r:] == 0)


class SCLDecoder:
    """SCL / CA-SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def _if_info(self):
        if_info = np.zeros(self.N, dtype=np.int32)
        if_info[self.info_indices] = 1
        return if_info

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float32)
        if_info = self._if_info()

        if self.crc_length > 0:
            crc_poly = (
                np.array([1, 0, 0, 0, 0, 0, 1, 1, 1], dtype=np.int32)
                if self.crc_length == 8
                else np.array(
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    dtype=np.int32,
                )
            )
            k_crc = len(self.info_indices) - self.crc_length
            u_hat = _SCLCRCDecoder(
                llr_ch,
                if_info,
                self.info_indices.astype(np.int32),
                k_crc,
                self.list_size,
                crc_poly,
            )
        else:
            u_hat = _SCLDecoder(llr_ch, if_info, self.list_size)

        return u_hat.astype(int), 0.0
