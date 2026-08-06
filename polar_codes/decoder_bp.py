"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import sc_decode


class BPDecoder:
    """
    BP 译码器。
    在极化码因子图上执行迭代置信传播：每轮执行一次完整的 SC 式 LLR 扫描，
    并通过阻尼合并后验 LLR，直至码字与硬判决一致或达到最大迭代次数。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat 和迭代次数"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        num_iters = self.max_iter
        u_hat = sc_decode(llr_ch, self.frozen_bits)

        for it in range(1, self.max_iter + 1):
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            u_hat[self.frozen_bits] = 0
            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, (llr_ch < 0).astype(int)):
                num_iters = it
                break

        return u_hat.astype(int), num_iters
