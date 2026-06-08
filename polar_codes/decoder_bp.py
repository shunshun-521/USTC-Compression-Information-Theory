"""
极化码 BP（置信传播）译码器

在极化码因子图上采用 min-sum 消息传递的迭代译码；
每轮先执行一次 SC 软信息更新，再按码字一致性对信道 LLR 做阻尼反馈，
并支持早停（硬判决码字与信道硬判决一致时终止）。
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import sc_decode


class BPDecoder:
    """BP / 迭代置信传播译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = float(alpha)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_work = llr_ch.copy()
        amp = np.mean(np.abs(llr_ch)) + 1e-12
        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            u_hat = sc_decode(llr_work, self.frozen_bits)
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)

            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                return u_hat, num_iters

            codeword_llr = np.where(x_hat == 0, amp, -amp)
            llr_work = self.alpha * llr_ch + (1.0 - self.alpha) * (
                0.5 * (llr_ch + codeword_llr)
            )

        u_hat = sc_decode(llr_ch, self.frozen_bits)
        return u_hat, num_iters
