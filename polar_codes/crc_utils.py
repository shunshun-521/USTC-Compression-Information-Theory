"""CRC-8 / CRC-16 按位 LFSR"""
import numpy as np


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = _crc_poly(crc_length)
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    for bit in info_bits:
        fb = ((reg >> (crc_length - 1)) ^ bit) & 1
        reg = (reg << 1) & mask
        if fb:
            reg ^= poly
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 crc 是否正确"""
    bits = np.asarray(bits, dtype=int).ravel()
    poly = _crc_poly(crc_length)
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in bits:
        fb = ((reg >> (crc_length - 1)) ^ bit) & 1
        reg = (reg << 1) & mask
        if fb:
            reg ^= poly
    return reg == 0
