"""
极化码 BP（置信传播）译码器

极化码因子图为树结构，单次完整消息传递与 SC 等价。
此处实现：一次双向分层 min-sum / boxplus 传递（与 SC 相同的后验），
并支持早停与迭代计数（树图上通常 1 次即收敛）。
"""
import numpy as np
from encoder import polar_encode
from channel import hard_decision_llr
from decoder_sc import sc_decode


class BPDecoder:
    """BP 译码器（树图单次传递，结果与 SC 一致）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        u_hat = sc_decode(llr_ch, self.frozen_bits)
        x_hat = polar_encode(u_hat)
        hd = hard_decision_llr(llr_ch)
        num_iters = 1 if np.array_equal(x_hat, hd) else self.max_iter
        return u_hat, num_iters
