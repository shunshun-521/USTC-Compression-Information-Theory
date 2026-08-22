"""CRC 辅助模块（供 internal SCL 使用）。"""
from crc_utils import crc_check


class CRC:
  def __init__(self, info_bits, crc_n):
    self.info_bits = list(info_bits)
    self.crc_n = crc_n

  def detection(self):
    if self.crc_n == 0:
      return 1
    return 1 if crc_check(self.info_bits, self.crc_n) else 0
