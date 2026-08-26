"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation


class BPDecoder:
    """
    BP 译码器。
    因子图 n+1 列（0=信源端，n=信道端），min-sum 修正因子 alpha。
    """

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(self.max_iter):
            num_iters = it + 1

            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        il, ir = i + k, i + k + step
                        L[il, j - 1] = self._f_ms(
                            R[il, j] + L[ir, j], L[il, j]
                        )
                        L[ir, j - 1] = self._f_ms(R[il, j], L[il, j]) + L[ir, j]

            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        il, ir = i + k, i + k + step
                        R[il, j + 1] = self._f_ms(
                            R[ir, j] + L[ir, j + 1], R[il, j]
                        )
                        R[ir, j + 1] = self._f_ms(R[il, j], L[il, j + 1]) + R[ir, j]

            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            x_hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                break

        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

        return u_hat, num_iters


if __name__ == "__main__":
    from channel import compute_llr, bpsk_modulate

    N = 32
    frozen = np.zeros(N, dtype=bool)
    frozen[:N // 2] = True
    u = np.zeros(N, dtype=int)
    u[~frozen] = 1
    llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.5)
    dec = BPDecoder(N, frozen, max_iter=50)
    u_hat, iters = dec.decode(llr)
    print("BP decode iters:", iters, "match:", np.array_equal(u, u_hat))
