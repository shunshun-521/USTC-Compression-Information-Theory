"""CRC 编码/校验"""
import numpy as np

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, width):
    reg = 0
    mask = (1 << width) - 1
    top = 1 << (width - 1)
    for bit in bits:
        msb = (reg >> (width - 1)) & 1
        reg = ((reg << 1) | int(bit)) & mask
        if msb:
            reg ^= poly
    for _ in range(width):
        msb = (reg >> (width - 1)) & 1
        reg = (reg << 1) & mask
        if msb:
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    if crc_length == 8:
        remainder = _crc_remainder(info_bits, _CRC8_POLY, 8)
        crc_bits = np.array([(remainder >> i) & 1 for i in range(7, -1, -1)], dtype=int)
    elif crc_length == 16:
        remainder = _crc_remainder(info_bits, _CRC16_POLY, 16)
        crc_bits = np.array([(remainder >> i) & 1 for i in range(15, -1, -1)], dtype=int)
    else:
        raise ValueError("crc_length must be 8 or 16")
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int).ravel()
    if crc_length == 8:
        remainder = _crc_remainder(bits, _CRC8_POLY, 8)
        return remainder == 0
    if crc_length == 16:
        remainder = _crc_remainder(bits, _CRC16_POLY, 16)
        return remainder == 0
    raise ValueError("crc_length must be 8 or 16")
