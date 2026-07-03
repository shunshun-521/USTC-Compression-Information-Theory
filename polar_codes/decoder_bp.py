"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from decoder_sc import f_operation
from encoder import polar_encode, bit_reversal_permutation


class BPDecoder:
    """BP 译码器（参考 Arikan 因子图结构）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self._large = 1e6

    def _g(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, num_iters)。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        # L[i,j], R[i,j]: i=0 为信源端, i=n 为信道端
        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)

        rev = bit_reversal_permutation(N)
        L[n, :] = llr_ch[rev]

        R[0, :] = 0.0
        R[0, self.frozen_bits] = self._large

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            # 从右到左更新 L
            for i in range(n - 1, -1, -1):
                half = 1 << i
                for j in range(half):
                    for k in range(0, N, 2 * half):
                        idx = k + j
                        L[i, idx] = self._g(
                            L[i + 1, idx], L[i + 1, idx + half] + R[i, idx + half]
                        )
                        L[i, idx + half] = self._g(R[i, idx], L[i + 1, idx]) + L[i + 1, idx + half]

            # 从左到右更新 R
            for i in range(1, n + 1):
                half = 1 << (i - 1)
                for j in range(half):
                    for k in range(0, N, 2 * half):
                        idx = k + j
                        R[i, idx] = self._g(
                            R[i, idx + half], L[i, idx + half] + R[i - 1, idx + half]
                        )
                        R[i, idx + half] = self._g(R[i - 1, idx], L[i, idx]) + R[i, idx + half]

            total = L[0, :] + R[0, :]
            u_hat = np.where(total >= 0, 0, 1).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        return u_hat, num_iters


if __name__ == "__main__":
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
    from construction import ga_construction

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    fb = frozen_bits.astype(bool)

    bp = BPDecoder(N, fb, max_iter=50)
    rng = np.random.default_rng(0)
    for sigma in [0.001, 0.1]:
        errors = 0
        for _ in range(20):
            u = np.zeros(N, dtype=int)
            u[info_idx] = rng.integers(0, 2, size=K)
            llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
            u_hat, iters = bp.decode(llr)
            if not np.array_equal(u_hat, u):
                errors += 1
        print(f"sigma={sigma} BP errors: {errors}/20")
