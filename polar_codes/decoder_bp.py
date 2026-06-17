"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode


def _minsum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * min(abs(a), abs(b))


class BPDecoder:
    """BP 译码器（因子图 min-sum）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self._large = 1e6

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, num_iters)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = [[0.0] * (n + 1) for _ in range(N)]
        R = [[0.0] * (n + 1) for _ in range(N)]

        for i in range(N):
            L[i][n] = llr_ch[i]
            R[i][0] = self._large if self.frozen_bits[i] else 0.0

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            # 右到左更新 L（列 n-1 到 0）
            for j in range(n - 1, -1, -1):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        Li = i + k
                        Lis = i + k + s
                        f_in = _minsum(R[Li][j] + L[Lis][j + 1], L[Li][j + 1], self.alpha)
                        L[Li][j] = f_in
                        L[Lis][j] = (
                            _minsum(R[Li][j], L[Li][j + 1], self.alpha) + L[Lis][j + 1]
                        )

            # 左到右更新 R（列 0 到 n-1）
            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        Li = i + k
                        Lis = i + k + s
                        R[Li][j + 1] = _minsum(
                            R[Lis][j] + L[Lis][j + 1], R[Li][j], self.alpha
                        )
                        R[Lis][j + 1] = (
                            _minsum(R[Li][j], L[Li][j + 1], self.alpha) + R[Lis][j]
                        )

            # 判决与早停
            for i in range(N):
                total = L[i][0] + R[i][0]
                u_hat[i] = 0 if total >= 0 else 1
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard):
                num_iters = it
                break

        return u_hat, num_iters


if __name__ == "__main__":
    from channel import bpsk_modulate, compute_llr
    from construction import ga_construction

    N = 64
    K = 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(2)
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, K)
    llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.5)

    bp = BPDecoder(N, frozen_bits)
    u_hat, iters = bp.decode(llr)
    print("BP correct:", np.array_equal(u_hat[info_idx], u[info_idx]), "iters:", iters)
