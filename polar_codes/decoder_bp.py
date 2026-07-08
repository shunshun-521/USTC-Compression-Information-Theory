"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.rev = bit_reversal_permutation(N)
        self.LARGE = 1e6

    def _minsum_f(self, a, b):
        return self.alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))

    def decode(self, llr_ch):
        """主译码函数。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        rev = self.rev
        llr = llr_ch[rev].copy()

        N, n = self.N, self.n
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            # 右到左更新 L
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        idx = i + k
                        L[idx, j - 1] = self._minsum_f(
                            R[idx, j] + L[idx + step, j],
                            L[idx, j],
                        )
                        L[idx + step, j - 1] = (
                            self._minsum_f(R[idx, j], L[idx, j])
                            + L[idx + step, j]
                        )

            # 左到右更新 R
            for j in range(1, n + 1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        idx = i + k
                        R[idx, j] = self._minsum_f(
                            R[idx + step, j] + L[idx + step, j],
                            R[idx, j - 1],
                        )
                        R[idx + step, j] = (
                            self._minsum_f(R[idx, j - 1], L[idx, j])
                            + R[idx + step, j]
                        )

            num_iters = it
            total = L[:, 0] + R[:, 0]
            u_nat = np.zeros(N, dtype=int)
            u_nat[~self.frozen_bits] = (total[~self.frozen_bits] < 0).astype(int)
            u_hat = u_nat

            x_hard = (llr_ch < 0).astype(int)
            x_reenc = polar_encode(u_hat)
            if np.array_equal(x_reenc, x_hard):
                break

        return u_hat, num_iters


if __name__ == "__main__":
    from construction import ga_construction
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    bp = BPDecoder(N, frozen_bits)
    sigma = eb_n0_to_sigma(6.0, K / N)
    ok = 0
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = np.random.randint(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)) + np.random.normal(0, sigma, N), sigma)
        uh, iters = bp.decode(llr)
        ok += np.array_equal(uh[info_idx], u[info_idx])
    print(f"BP test: {ok}/20")
