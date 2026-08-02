"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from _scl_core import scl_tree_decode
from encoder import _bit_rev_indices


CRC_POLYNOMIALS = {8: 0x07, 16: 0x8005}


def crc_encode(info_bits, crc_length=8):
  """计算 CRC 校验位并附加到信息比特后。"""
  poly = CRC_POLYNOMIALS[crc_length]
  info_bits = np.asarray(info_bits, dtype=int)
  reg = 0
  for bit in info_bits:
    reg <<= 1
    reg |= int(bit)
    if reg & (1 << crc_length):
      reg ^= poly
  crc_bits = np.array(
    [(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int
  )
  return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
  """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
  poly = CRC_POLYNOMIALS[crc_length]
  bits = np.asarray(bits, dtype=int)
  reg = 0
  for bit in bits:
    reg <<= 1
    reg |= int(bit)
    if reg & (1 << crc_length):
      reg ^= poly
  return reg == 0


class SCLDecoder:
  """SCL 译码器。"""

  def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
    self.N = N
    self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
    self.list_size = list_size
    self.crc_length = crc_length
    self.info_indices = np.where(~self.frozen_bits)[0]

  def decode(self, llr_ch):
    """主译码函数。"""
    brp = _bit_rev_indices(self.N)
    llr_br = np.asarray(llr_ch, dtype=np.float64)[brp]
    crc_fn = None
    if self.crc_length > 0:
      crc_fn = lambda bits: crc_check(bits, self.crc_length)
    u_hat, pm = scl_tree_decode(
      llr_br, self.info_indices, self.list_size, crc_fn
    )
    u_hat[self.frozen_bits] = 0
    return u_hat, pm
