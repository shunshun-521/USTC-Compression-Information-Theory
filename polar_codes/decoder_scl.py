"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import _SCLCore

# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, crc_length):
    bits = np.asarray(bits, dtype=np.int8).astype(int)
    if crc_length == 8:
        poly = _CRC8_POLY
        width = 8
    elif crc_length == 16:
        poly = _CRC16_POLY
        width = 16
    else:
        raise ValueError("crc_length must be 8 or 16")
    reg = 0
    for b in bits:
        reg ^= b << (width - 1)
        for _ in range(8):
            if reg & (1 << (width - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << width) - 1)
            else:
                reg = (reg << 1) & ((1 << width) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    rem = _crc_remainder(info_bits, crc_length)
    crc_bits = np.array(
        [(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    if crc_length == 0:
        return True
    payload = bits[:-crc_length]
    expected = crc_encode(payload, crc_length)
    return np.array_equal(bits, expected)


class SCLDecoder:
    """SCL 译码器（含 CRC 辅助 CA-SCL）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_idx = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = _SCLCore(self.frozen_bits, self.list_size).decode(llr_ch)

        if self.crc_length > 0:
            for pm, u_hat in paths:
                info_bits = u_hat[self.info_idx]
                if crc_check(info_bits, self.crc_length):
                    return u_hat.astype(int), pm

        best_pm, best_u = paths[0]
        return best_u.astype(int), best_pm
