"""
极化码 BP（置信传播）译码器
基于校验矩阵 H 的 min-sum BP，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode


def _build_generator(n):
    F = np.array([[1, 1], [0, 1]], dtype=int)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    return G % 2


def _f_min_sum(a, b, alpha=0.9375):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（在校验矩阵 H 上对码字位运行 min-sum BP）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

        G = _build_generator(self.n)
        self.Ginv = np.linalg.inv(G.astype(float)).astype(int) % 2
        frozen_idx = np.where(self.frozen_bits)[0]
        self.H = self.Ginv[frozen_idx, :]
        self.M = len(frozen_idx)

    def _codeword_to_source(self, x_hat):
        u_hat = (self.Ginv @ x_hat) % 2
        u_hat[self.frozen_bits] = 0
        return u_hat.astype(int)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, M, alpha = self.N, self.M, self.alpha

        q = np.zeros((M, N), dtype=np.float64)
        r = np.zeros((M, N), dtype=np.float64)
        active = self.H.astype(bool)
        q[active] = llr_ch[np.tile(np.arange(N), (M, 1))[active]]

        num_iters = 0
        x_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for m in range(M):
                idxs = np.where(self.H[m])[0]
                if len(idxs) == 0:
                    continue
                msgs = q[m, idxs] - r[m, idxs]
                for k, vn in enumerate(idxs):
                    others = np.delete(msgs, k)
                    if len(others) == 0:
                        r[m, vn] = 0.0
                    else:
                        prod_sign = np.prod(np.sign(others))
                        min_abs = np.min(np.abs(others))
                        r[m, vn] = alpha * prod_sign * min_abs

            for n in range(N):
                cns = np.where(self.H[:, n])[0]
                total = llr_ch[n] + np.sum(r[cns, n])
                for m in cns:
                    q[m, n] = total - r[m, n]

            num_iters = it

            for n in range(N):
                total = llr_ch[n] + np.sum(r[:, n])
                x_hat[n] = 0 if total >= 0 else 1

            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        for n in range(N):
            total = llr_ch[n] + np.sum(r[:, n])
            x_hat[n] = 0 if total >= 0 else 1

        u_hat = self._codeword_to_source(x_hat)
        return u_hat, num_iters
