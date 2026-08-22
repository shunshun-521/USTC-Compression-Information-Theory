"""CRC 工具函数。"""
import numpy as np

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_process(bits, poly, crc_length):
  reg = 0
  mask = (1 << crc_length) - 1
  top = 1 << (crc_length - 1)
  for bit in bits:
    reg ^= int(bit) << (crc_length - 1)
    if reg & top:
      reg = ((reg << 1) ^ poly) & mask
    else:
      reg = (reg << 1) & mask
  return reg


def crc_encode(info_bits, crc_length=8):
  info_bits = np.asarray(info_bits, dtype=int)
  poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
  remainder = _crc_process(info_bits, poly, crc_length)
  crc_bits = np.array(
    [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
    dtype=int,
  )
  return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
  bits = np.asarray(bits, dtype=int)
  poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
  return _crc_process(bits, poly, crc_length) == 0
