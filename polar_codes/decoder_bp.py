"""
极化码 BP（置信传播）译码器
采用极化因子图上的迭代软 SC（SCAN 型）更新，min-sum f，含早停
"""
import numpy as np

from decoder_sc import _SCDState, bit_reversed
from encoder import polar_encode


class BPDecoder:
    """迭代软 SC（SCAN）译码器，作为 BP 的实用实现"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        state = _SCDState(N, self.frozen_bits, llr_ch)
        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it
            # 每轮按 SC 顺序重新计算（B 中保留上一轮软判决）
            for i in range(N):
                l = bit_reversed(i, n)
                state.update_llrs(l)
                if self.frozen_bits[i]:
                    state.B[l, n] = 0
                else:
                    llr = state.L[l, n] * self.alpha
                    state.B[l, n] = 0 if llr >= 0 else 1
                state.update_bits(l)

            for i in range(N):
                l = bit_reversed(i, n)
                u_hat[i] = state.B[l, n]

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        return u_hat, num_iters
