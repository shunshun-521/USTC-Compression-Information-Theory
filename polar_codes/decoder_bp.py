"""
极化码 BP（置信传播）译码器
基于因子图的 min-sum BP，含早停
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation

LARGE = 1e8


def _minsum_f(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """
    BP 译码器。
    采用与 SC 相同的 L[:,0] 信道列布局，在蝶形因子上迭代传递软信息。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        rev = bit_reversal_permutation(N)
        self.decode_order = [int(rev[i]) for i in range(N)]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        alpha = self.alpha

        # 软信息树：与 SC 相同的 (N, n+1) 布局
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, 0] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            # 自上而下（f/g 正向，软 SC 一步）
            for s in range(n):
                block = 1 << (s + 1)
                half = block >> 1
                for base in range(0, N, block):
                    for k in range(half):
                        i = base + k
                        j = base + k + half
                        L[i, s + 1] = _minsum_f(L[i, s], L[j, s], alpha)
                        R[j, s + 1] = _minsum_f(L[i, s], R[i, s], alpha) + R[j, s]
                        L[j, s + 1] = _minsum_f(R[i, s], L[i, s], alpha) + L[j, s]

            # 自下而上（反向软信息）
            for s in range(n - 1, -1, -1):
                block = 1 << (s + 1)
                half = block >> 1
                for base in range(0, N, block):
                    for k in range(half):
                        i = base + k
                        j = base + k + half
                        R[i, s] = _minsum_f(R[i, s + 1], R[j, s + 1], alpha)
                        L[j, s] = _minsum_f(L[i, s + 1], L[j, s + 1], alpha) + L[j, s]

            for i in range(N):
                total = L[i, n] + R[i, n]
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        for i in range(N):
            total = L[i, n] + R[i, n]
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat.astype(int), num_iters
