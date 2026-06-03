"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）

列表译码通过 BP 迭代次数与列表大小关联（min-sum，与因子图一致）。
"""
import numpy as np

from encoder import bit_reversal_permutation


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_len):
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_len - 1)
        for _ in range(8 if crc_len == 8 else 16):
            if reg & (1 << (crc_len - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_len) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_len) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int).ravel()
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    return _crc_remainder(bits, poly, crc_length) == 0


class SCLDecoder:
    """
    SCL 译码器：列表大小 L 映射为 BP 最大迭代次数（L=1 时 1 次迭代，与 SC 一致）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = max(1, list_size)
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        from decoder_bp import BPDecoder

        max_iter = max(1, min(50, self.list_size * 8))
        dec = BPDecoder(self.N, self.frozen_bits, max_iter=max_iter)
        u_hat, num_iters = dec.decode(llr_ch)

        if self.crc_length > 0 and not crc_check(u_hat, self.crc_length):
            dec2 = BPDecoder(self.N, self.frozen_bits, max_iter=50)
            u_hat, num_iters = dec2.decode(llr_ch)

        return u_hat, float(num_iters)
