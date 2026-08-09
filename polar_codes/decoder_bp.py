"""
极化码 BP（置信传播）译码器
基于极化码校验矩阵 Tanner 图，min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, build_generator_matrix, gf2_inverse


def _ms_g(a, b, alpha):
    """min-sum 近似。"""
    sa = np.sign(a)
    sb = np.sign(b)
    sa = sa if sa != 0 else 1.0
    sb = sb if sb != 0 else 1.0
    return alpha * sa * sb * min(abs(a), abs(b))


class BPDecoder:
    """BP 译码器（Tanner 图 min-sum）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.875):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

        G = build_generator_matrix(N)
        Ginv = gf2_inverse(G)
        frozen_idx = np.where(self.frozen_bits)[0]
        self.H = Ginv[frozen_idx, :]
        self.Ginv = Ginv

        self._chk_edges = []
        for m in range(self.H.shape[0]):
            self._chk_edges.append(np.where(self.H[m])[0])

    def _min_sum_bp(self, llr_ch):
        """在码字空间执行 min-sum BP。"""
        M, N = self.H.shape
        R = np.zeros((M, N), dtype=np.float64)
        Lq = llr_ch.copy()
        alpha = self.alpha

        for _ in range(self.max_iter):
            for m, idx in enumerate(self._chk_edges):
                if len(idx) == 0:
                    continue
                qvals = {n: Lq[n] - R[m, n] for n in idx}
                for n in idx:
                    others = [j for j in idx if j != n]
                    if not others:
                        R[m, n] = 0.0
                        continue
                    sign = 1.0
                    min_abs = np.inf
                    for j in others:
                        v = qvals[j]
                        if v < 0:
                            sign *= -1.0
                        min_abs = min(min_abs, abs(v))
                    R[m, n] = alpha * sign * min_abs

            for n in range(N):
                Lq[n] = llr_ch[n] + np.sum(R[:, n])

        return (Lq < 0).astype(int)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, num_iters)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        num_iters = 0
        u_hat = np.zeros(self.N, dtype=int)

        M, N = self.H.shape
        R = np.zeros((M, N), dtype=np.float64)
        Lq = llr_ch.copy()
        alpha = self.alpha

        for it in range(self.max_iter):
            num_iters = it + 1

            for m, idx in enumerate(self._chk_edges):
                if len(idx) == 0:
                    continue
                qvals = {n: Lq[n] - R[m, n] for n in idx}
                for n in idx:
                    others = [j for j in idx if j != n]
                    if not others:
                        R[m, n] = 0.0
                        continue
                    sign = 1.0
                    min_abs = np.inf
                    for j in others:
                        v = qvals[j]
                        if v < 0:
                            sign *= -1.0
                        min_abs = min(min_abs, abs(v))
                    R[m, n] = alpha * sign * min_abs

            for n in range(N):
                Lq[n] = llr_ch[n] + np.sum(R[:, n])

            x_hat = (Lq < 0).astype(int)
            u_hat = (x_hat @ self.Ginv) % 2
            u_hat[self.frozen_bits] = 0

            x_enc = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_enc, hard_ch):
                break

        return u_hat.astype(int), num_iters
