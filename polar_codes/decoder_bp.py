"""
极化码 BP（置信传播）译码器
基于因子图的 min-sum 消息传递；采用迭代 SC-LLR 反馈增强收敛（早停）
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import sc_decode, f_operation


def _bp_f(x, y, alpha):
    return alpha * f_operation(x, y)


class BPDecoder:
    """
    BP 译码器。
    每轮执行因子图 min-sum 左右消息传递，并以 SC 译码结果做 LLR 反馈直至早停。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.max_iter = max_iter
        self.alpha = alpha
        fb = np.asarray(frozen_bits).reshape(-1).astype(int)
        self.frozen = fb == 1
        self.frozen_idx = np.where(self.frozen)[0]

    def _message_pass(self, L, R):
        """一轮 min-sum 左右消息传递"""
        n, N, alpha = self.n, self.N, self.alpha
        for s in range(n - 1, -1, -1):
            step = 1 << s
            for i in range(0, N, 2 * step):
                L[s][i] = _bp_f(
                    R[s + 1][i] + L[s + 1][i + step], L[s + 1][i], alpha
                )
                L[s][i + step] = (
                    _bp_f(R[s + 1][i], L[s + 1][i], alpha) + L[s + 1][i + step]
                )
        for s in range(n):
            step = 1 << s
            for i in range(0, N, 2 * step):
                R[s + 1][i] = _bp_f(
                    R[s][i + step] + L[s + 1][i + step], R[s][i], alpha
                )
                R[s + 1][i + step] = (
                    _bp_f(R[s][i], L[s + 1][i], alpha) + R[s][i + step]
                )

    def _hard_decision(self, L, R):
        post = L[0] + R[0]
        u_hat = (post < 0).astype(int)
        u_hat[self.frozen] = 0
        return u_hat

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N
        LARGE = 1e8

        L = [np.zeros(N) for _ in range(n + 1)]
        R = [np.zeros(N) for _ in range(n + 1)]
        llr_work = llr_ch.copy()
        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            L[n][:] = llr_work
            R[0][:] = 0.0
            for idx in self.frozen_idx:
                R[0][idx] = -LARGE if L[n][idx] >= 0 else LARGE

            self._message_pass(L, R)
            u_bp = self._hard_decision(L, R)
            u_sc = sc_decode(llr_work, self.frozen)

            x_bp = polar_encode(u_bp)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_bp, hard_ch):
                return u_bp, it

            x_sc = polar_encode(u_sc)
            if np.array_equal(x_sc, hard_ch):
                return u_sc, it

            # LLR 反馈：将码字软信息叠加到信道 LLR
            llr_work = llr_ch + self.alpha * (1 - 2 * u_sc) * np.abs(llr_ch)
            num_iters = it

        return sc_decode(llr_work, self.frozen), num_iters
