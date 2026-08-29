"""
极化码 BP（置信传播）译码器
基于校验矩阵的 min-sum BP，含早停机制
"""
import numpy as np
from encoder import polar_encode, build_generator_matrix
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器（校验矩阵 Tanner 图，变量节点为码字比特）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_indices = np.where(self.frozen_bits)[0]
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha

        G = build_generator_matrix(N)
        G_inv = self._invert_binary_matrix(G)
        self.G_inv = G_inv
        self.H = G_inv[:, self.frozen_indices].T.astype(np.int8)

        self.cn_neighbors = [np.where(self.H[m])[0] for m in range(self.H.shape[0])]
        self.vn_neighbors = [np.where(self.H[:, v])[0] for v in range(N)]

    @staticmethod
    def _invert_binary_matrix(M):
        n = M.shape[0]
        A = np.concatenate([M.astype(np.int8), np.eye(n, dtype=np.int8)], axis=1)
        for col in range(n):
            pivot = next(r for r in range(col, n) if A[r, col])
            if pivot != col:
                A[[col, pivot]] = A[[pivot, col]]
            for row in range(n):
                if row != col and A[row, col]:
                    A[row] ^= A[col]
        return A[:, n:]

    def _cn_update(self, msgs):
        if len(msgs) <= 1:
            return 0.0
        abs_msgs = np.abs(msgs)
        min_idx = np.argmin(abs_msgs)
        min_val = abs_msgs[min_idx]
        sign = np.prod(np.sign(msgs))
        second_min = np.min(np.delete(abs_msgs, min_idx))
        out = []
        for i, m in enumerate(msgs):
            s = sign * np.sign(m) if m != 0 else 0.0
            mag = second_min if i == min_idx else min_val
            out.append(self.alpha * s * mag)
        return out

    def _x_to_u(self, x_hat):
        u_hat = (x_hat @ self.G_inv) % 2
        u_hat[self.frozen_indices] = 0
        return u_hat.astype(int)

    def decode(self, llr_ch):
        N = self.N
        M = self.H.shape[0]
        ch_llr = np.asarray(llr_ch, dtype=np.float64)

        Lr = np.zeros((M, N), dtype=np.float64)
        num_iters = 0

        for it in range(1, self.max_iter + 1):
            Lq = np.zeros((M, N), dtype=np.float64)
            for v in range(N):
                total = ch_llr[v] + sum(Lr[m, v] for m in self.vn_neighbors[v])
                for m in self.vn_neighbors[v]:
                    Lq[m, v] = total - Lr[m, v]

            for m in range(M):
                nbrs = list(self.cn_neighbors[m])
                msgs = [Lq[m, v] for v in nbrs]
                out = self._cn_update(msgs)
                for v, val in zip(nbrs, out):
                    Lr[m, v] = val

            x_hat = np.zeros(N, dtype=int)
            for v in range(N):
                total = ch_llr[v] + sum(Lr[m, v] for m in self.vn_neighbors[v])
                x_hat[v] = 0 if total >= 0 else 1

            hard_ch = (ch_llr < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break
            num_iters = it

        x_hat = np.zeros(N, dtype=int)
        for v in range(N):
            total = ch_llr[v] + sum(Lr[m, v] for m in self.vn_neighbors[v])
            x_hat[v] = 0 if total >= 0 else 1

        u_hat = self._x_to_u(x_hat)
        return u_hat, num_iters
