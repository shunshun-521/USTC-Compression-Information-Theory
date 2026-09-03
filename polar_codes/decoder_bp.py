"""
极化码 BP（置信传播）译码器
基于对偶约束 H·x=0 的 min-sum BP（变量节点为码字比特），含早停机制
"""
import math
import numpy as np
from encoder import polar_generator_matrix
from channel import hard_decision_llr


def _gf2_inverse(G):
    N = G.shape[0]
    A = np.hstack([G.astype(int), np.eye(N, dtype=int)]) % 2
    for col in range(N):
        pivot = next(r for r in range(col, N) if A[r, col])
        if pivot != col:
            A[[col, pivot]] = A[[pivot, col]]
        for r in range(N):
            if r != col and A[r, col]:
                A[r] = (A[r] + A[col]) % 2
    return A[:, N:]


class BPDecoder:
    """基于校验矩阵 H·x=0 的 min-sum BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        G = polar_generator_matrix(N)
        G_inv = _gf2_inverse(G)
        frozen_idx = np.where(self.frozen_bits)[0]
        self.H = G_inv[frozen_idx, :].astype(int)
        self.G_inv = G_inv
        self.M = self.H.shape[0]
        self.cn_neighbors = [np.where(self.H[m])[0] for m in range(self.M)]
        self.vn_neighbors = [np.where(self.H[:, n])[0] for n in range(N)]
        self._cn_deg = np.array([len(x) for x in self.cn_neighbors], dtype=int)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, M, alpha = self.N, self.M, self.alpha

        R = np.zeros((M, N), dtype=np.float64)
        num_iters = self.max_iter
        L_post = llr_ch.copy()

        for it in range(1, self.max_iter + 1):
            L_v = llr_ch + R.sum(axis=0)

            for m in range(M):
                idx = self.cn_neighbors[m]
                q = L_v[idx] - R[m, idx]
                abs_q = np.abs(q)
                signs = np.sign(q + 1e-12)
                global_sign = np.prod(signs)
                min1 = np.min(abs_q)
                min2 = np.partition(abs_q, 1)[1] if len(q) > 1 else min1
                for k, n in enumerate(idx):
                    use_min = min2 if abs_q[k] == min1 and len(q) > 1 else min1
                    R[m, n] = alpha * (global_sign / (signs[k] + 1e-12)) * use_min

            L_post = llr_ch + R.sum(axis=0)
            x_hat = (L_post < 0).astype(int)

            if not np.any((self.H @ x_hat) % 2):
                if np.array_equal(x_hat, hard_decision_llr(llr_ch)):
                    num_iters = it
                    break

        x_hat = (L_post < 0).astype(int)
        u_hat = (self.G_inv @ x_hat) % 2
        u_hat[self.frozen_bits] = 0

        return u_hat, num_iters
