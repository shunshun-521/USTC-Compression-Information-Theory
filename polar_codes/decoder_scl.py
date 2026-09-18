"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import sc_decode
from utils import crc_check, crc_encode


class SCLDecoder:
    """SCL 译码器（基于 SC 状态复制）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.info_indices = np.where(self.frozen_bits == 0)[0]
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0].tolist())
        self.list_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        from decoder_scl_core import scl_decode_core

        u_hat, pm = scl_decode_core(
            llr_ch,
            self.info_indices,
            self.frozen_set,
            self.list_size,
            self.crc_length,
        )
        return u_hat, pm
