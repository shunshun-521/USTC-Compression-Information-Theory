"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import sc_decode
from internal.pc_decoder import scl_decoder as _scl_decoder_raw

from crc_utils import crc_encode, crc_check

class SCLDecoder:
  """SCL 译码器。"""

  def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
    self.N = N
    self.n = int(np.log2(N))
    self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
    self.info_indices = np.where(~self.frozen_bits)[0]
    self.list_size = list_size
    self.crc_length = crc_length
    self.rev = bit_reversal_permutation(N)

  def decode(self, llr_ch):
    """主译码函数，返回 (u_hat, pm)。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)

    if self.list_size == 1 and self.crc_length == 0:
      u_hat = sc_decode(llr_ch, self.frozen_bits)
      return u_hat, 0.0

    info_list = list(self.info_indices)
    llr_in = llr_ch[self.rev]
    decode_para = [self.list_size, 'hf']

    if self.crc_length > 0:
      u_hat = _scl_decoder_raw(
        llr_in, info_list, 0, decode_para, self.crc_length,
      )
    else:
      u_hat = _scl_decoder_raw(
        llr_in, info_list, 0, decode_para, 0,
      )

    return u_hat.astype(int), 0.0
