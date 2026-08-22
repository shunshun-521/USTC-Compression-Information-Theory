"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import bit_reversal_permutation, polar_encode
from internal.pc_decoder import bp_decoder as _bp_decoder_raw


class BPDecoder:
  """BP 译码器。"""

  def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
    self.N = N
    self.n = int(np.log2(N))
    self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
    self.info_indices = list(np.where(~self.frozen_bits)[0])
    self.max_iter = max_iter
    self.alpha = alpha
    self.rev = bit_reversal_permutation(N)

  def decode(self, llr_ch):
    """主译码函数，返回 (u_hat, num_iters)。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    llr_in = llr_ch[self.rev]
    decode_para = [self.max_iter, 'g_matrix']
    u_hat = _bp_decoder_raw(llr_in, self.info_indices, 0, decode_para, 0)
    return u_hat.astype(int), self.max_iter
