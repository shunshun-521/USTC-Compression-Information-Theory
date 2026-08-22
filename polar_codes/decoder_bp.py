"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode
from decoder_sc import align_llr_for_decoder


def _f_min_sum(a, b, alpha):
    """min-sum f 运算，带修正因子 alpha。"""
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """
    BP 译码器。
    因子图有 n+1 列（列 0 到列 n），列 0 为信源端，列 n 为信道端。
    """

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, num_iters
        """
        N, n = self.N, self.n
        llr_raw = np.asarray(llr_ch, dtype=np.float64)
        llr_ch = align_llr_for_decoder(llr_raw)

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        for i in range(N):
            if self.frozen_bits[i]:
                R[i, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(self.max_iter):
            num_iters = it + 1

            # 从右到左更新 L（层 n-1 .. 0）
            for lam in range(n - 1, -1, -1):
                half = 2 ** lam
                block = 2 * half
                for block_start in range(0, N, block):
                    for i in range(block_start, block_start + half):
                        j = i + half
                        L[i, lam] = _f_min_sum(
                            R[i, lam + 1] + L[j, lam + 1], L[i, lam + 1], self.alpha
                        )
                        L[j, lam] = _f_min_sum(
                            R[i, lam + 1], L[i, lam + 1], self.alpha
                        ) + L[j, lam + 1]

            # 从左到右更新 R（层 1 .. n-1）
            for lam in range(1, n):
                half = 2 ** (lam - 1)
                block = 2 * half
                for block_start in range(0, N, block):
                    for i in range(block_start, block_start + half):
                        j = i + half
                        R[i, lam] = _f_min_sum(
                            R[j, lam] + L[j, lam + 1], R[i, lam - 1], self.alpha
                        )
                        R[j, lam] = _f_min_sum(
                            R[i, lam - 1], L[i, lam + 1], self.alpha
                        ) + R[j, lam]

            # 早停检查
            for i in range(N):
                total = L[i, 0] + R[i, 0]
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_raw < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        for i in range(N):
            total = L[i, 0] + R[i, 0]
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat, num_iters
