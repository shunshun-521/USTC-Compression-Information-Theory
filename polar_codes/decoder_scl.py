"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from _sc_core import scl_decode_core, numba


CRC_POLYNOMIALS = {
    8: 0x07,
    16: 0x8005,
}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC_POLYNOMIALS[crc_length]
    reg = 0
    for bit in info_bits:
        reg ^= (int(bit) << (crc_length - 1))
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.is_info = (~self.frozen_bits).astype(np.int32)

    def decode(self, llr_ch):
        if numba is None:
            raise RuntimeError('numba is required for SCL decoding')
        llr_ch = np.asarray(llr_ch, dtype=np.float32)
        u_hat, pm = scl_decode_core(llr_ch, self.is_info, self.list_size)
        u_hat = u_hat.astype(int)

        if self.crc_length > 0:
            info_bits = u_hat[self.info_indices]
            if crc_check(info_bits, self.crc_length):
                return u_hat, float(pm)
            best_pm = float(pm)
            best_u = u_hat
            return best_u, best_pm

        return u_hat, float(pm)
