"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode, bit_reversal_permutation


def _f_min_sum(a, b, alpha):
    sign = np.sign(a) * np.sign(b)
    sign[sign == 0] = 1.0
    return alpha * sign * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)
        self.LARGE = 1e6

    def _hard_bits_from_llr(self, llr):
        bits = (llr < 0).astype(int)
        bits[self.frozen_bits] = 0
        return bits

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n
        alpha = self.alpha

        br = self.br
        llr_nat = np.zeros(N, dtype=np.float64)
        llr_nat[br] = llr_ch

        # L[i][j], R[i][j]: i=行(比特位置), j=列(层)
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_nat
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            # 从右到左更新 L
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for block in range(0, N, 2 * s):
                    left = slice(block, block + s)
                    right = slice(block + s, block + 2 * s)
                    L[left, j - 1] = _f_min_sum(
                        R[left, j] + L[right, j], L[left, j], alpha
                    )
                    L[right, j - 1] = _f_min_sum(R[left, j], L[left, j], alpha) + L[right, j]

            # 从左到右更新 R
            for j in range(0, n):
                s = 1 << j
                for block in range(0, N, 2 * s):
                    left = slice(block, block + s)
                    right = slice(block + s, block + 2 * s)
                    R[left, j + 1] = _f_min_sum(
                        R[right, j] + L[right, j + 1], R[left, j], alpha
                    )
                    R[right, j + 1] = _f_min_sum(R[left, j], L[left, j + 1], alpha) + R[right, j]

            total = L[:, 0] + R[:, 0]
            u_hat = self._hard_bits_from_llr(total)

            # 早停：重编码后与信道硬判决一致
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = self._hard_bits_from_llr(total)
        return u_hat.astype(int), num_iters


if __name__ == "__main__":
    from construction import ga_construction
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(2)
    bp = BPDecoder(N, frozen_bits)
    ok = 0
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        from encoder import polar_encode

        x = polar_encode(u)
        sigma = eb_n0_to_sigma(6.0, K / N)
        llr = compute_llr(bpsk_modulate(x) + rng.normal(0, sigma, N), sigma)
        u_hat, iters = bp.decode(llr)
        if np.array_equal(u, u_hat):
            ok += 1
    print(f"BP test: {ok}/20 correct at Eb/N0=6dB")
