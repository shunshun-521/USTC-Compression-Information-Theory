"""
极化码 BP（置信传播）译码器
基于奇偶校验矩阵 H 的 LDPC 风格 min-sum BP，含早停机制
"""
import numpy as np
from encoder import polar_encode


def _polar_G(N):
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = np.array([[1]], dtype=int)
    while G.shape[0] < N:
        G = np.kron(G, F)
    return G


def _build_parity_matrix(N, frozen_bits):
    frozen_idx = np.where(frozen_bits)[0]
    G = _polar_G(N)
    return G[:, frozen_idx].T.astype(np.float64)


class BPDecoder:
    """BP 译码器（LDPC min-sum on H = G^T[frozen rows]）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.H = _build_parity_matrix(N, self.frozen_bits)
        self.G = _polar_G(N)
        self.num_checks, _ = self.H.shape
        self.cn_edges = [np.where(self.H[c] == 1)[0] for c in range(self.num_checks)]
        self.vn_edges = [np.where(self.H[:, v] == 1)[0] for v in range(N)]

    def _min_sum_check_to_var(self, q_in, exclude_v, check_idx):
        edges = self.cn_edges[check_idx]
        signs = 1.0
        mins = np.inf
        min2 = np.inf
        for v in edges:
            if v == exclude_v:
                continue
            val = q_in[v]
            s = 1.0 if val >= 0 else -1.0
            if val == 0:
                s = 1.0
            signs *= s
            av = abs(val)
            if av < mins:
                min2 = mins
                mins = av
            elif av < min2:
                min2 = av
        if not np.isfinite(mins):
            return 0.0
        if not np.isfinite(min2):
            min2 = mins
        return self.alpha * signs * mins

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        L_ch = llr_ch.copy()

        q_cv = np.zeros((self.num_checks, N), dtype=np.float64)
        r_vc = np.zeros((self.num_checks, N), dtype=np.float64)

        num_iters = self.max_iter
        x_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            L_v = L_ch.copy()
            for c in range(self.num_checks):
                for v in self.cn_edges[c]:
                    L_v[v] += q_cv[c, v]

            for c in range(self.num_checks):
                for v in self.cn_edges[c]:
                    r_vc[c, v] = L_v[v] - q_cv[c, v]

            for c in range(self.num_checks):
                for v in self.cn_edges[c]:
                    q_cv[c, v] = self._min_sum_check_to_var(r_vc[c], v, c)

            L_total = L_ch.copy()
            for c in range(self.num_checks):
                for v in self.cn_edges[c]:
                    L_total[v] += q_cv[c, v]

            x_hat = (L_total < 0).astype(int)
            u_hat = (x_hat @ self.G) % 2
            u_hat[self.frozen_bits] = 0

            x_reenc = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_reenc, hard_ch):
                num_iters = it
                break

        L_total = L_ch.copy()
        for c in range(self.num_checks):
            for v in self.cn_edges[c]:
                L_total[v] += q_cv[c, v]
        x_hat = (L_total < 0).astype(int)
        u_hat = (x_hat @ self.G) % 2
        u_hat[self.frozen_bits] = 0

        return u_hat, num_iters
