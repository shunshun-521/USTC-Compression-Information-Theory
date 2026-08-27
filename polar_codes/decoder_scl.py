"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import torch
from sionna.phy.fec.polar import PolarSCLDecoder

from polar_core import sc_decode

_CRC_MAP = {8: "CRC6", 16: "CRC16"}


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError(f"Unsupported CRC length: {crc_length}")
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    if crc_length == 0:
        return True
    return np.array_equal(bits[-crc_length:], crc_encode(bits[:-crc_length], crc_length))


class SCLDecoder:
    """SCL 译码器（L=1 使用自研 SC，L>1 使用 Sionna cpu_only SCL 核心）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]
        self.frozen_pos = np.where(self.frozen_bits == 1)[0]
        self._sionna_dec = None
        if list_size > 1:
            crc_degree = _CRC_MAP.get(crc_length)
            self._sionna_dec = PolarSCLDecoder(
                self.frozen_pos,
                N,
                list_size=list_size,
                crc_degree=crc_degree,
                cpu_only=True,
            )

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr_torch = torch.tensor(-llr_ch, dtype=torch.float32).unsqueeze(0)
        u_info = self._sionna_dec(llr_torch).numpy()[0]
        u_hat = np.zeros(self.N, dtype=int)
        u_hat[self.info_indices] = np.round(u_info).astype(int)

        if self.crc_length > 0:
            info_bits = u_hat[self.info_indices]
            if not crc_check(info_bits, self.crc_length):
                pass

        return u_hat, 0.0
