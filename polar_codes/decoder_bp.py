"""
极化码 BP（置信传播）译码器
基于冻结位导出的校验矩阵 H 的稀疏 min-sum BP
"""
import numpy as np
import math
from encoder import polar_encode, generator_matrix


class BPDecoder:
    """BP 译码器（稀疏边消息传递）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        if 2**self.n != N:
            raise ValueError(f"N={N} must be a power of 2")
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self._large = 20.0

        G = generator_matrix(N)
        H = G[:, np.where(self.frozen_bits)[0]].T.astype(np.int8)
        self.M = H.shape[0]

        # 稀疏边列表 (m, v)
        self._cn_edges = [np.where(H[m])[0] for m in range(self.M)]
        self._vn_edges = [np.where(H[:, v])[0] for v in range(N)]
        self._edge_mv = [(m, v) for m in range(self.M) for v in self._cn_edges[m]]

    def decode(self, llr_ch):
        llr_ch = np.clip(np.asarray(llr_ch, dtype=np.float64), -self._large, self._large)
        N, M = self.N, self.M

        r_mv = {e: 0.0 for e in self._edge_mv}
        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            q_mv = {}
            for m, v in self._edge_mv:
                q_mv[(m, v)] = llr_ch[v] + sum(
                    r_mv[(m2, v)] for m2 in self._vn_edges[v] if m2 != m
                )
                q_mv[(m, v)] = float(np.clip(q_mv[(m, v)], -self._large, self._large))

            for m in range(M):
                nbrs = self._cn_edges[m]
                for v in nbrs:
                    sign = 1.0
                    mag = self._large
                    for o in nbrs:
                        if o == v:
                            continue
                        val = q_mv[(m, o)]
                        sign *= np.sign(val) if val != 0 else 1.0
                        mag = min(mag, abs(val))
                    r_mv[(m, v)] = self.alpha * sign * mag

            total = np.zeros(N, dtype=np.float64)
            for v in range(N):
                total[v] = llr_ch[v] + sum(r_mv[(m, v)] for m in self._vn_edges[v])
            x_hat = (total < 0).astype(np.int8)
            u_hat = (x_hat @ generator_matrix(N)) % 2
            u_hat[self.frozen_bits] = 0

            if self._early_stop(u_hat, llr_ch):
                num_iters = it
                break

        total = np.zeros(N, dtype=np.float64)
        for v in range(N):
            total[v] = llr_ch[v] + sum(r_mv[(m, v)] for m in self._vn_edges[v])
        x_hat = (total < 0).astype(np.int8)
        u_hat = (x_hat @ generator_matrix(N)) % 2
        u_hat[self.frozen_bits] = 0
        return u_hat.astype(np.int8), num_iters

    def _early_stop(self, u_hat, llr_ch):
        x_hat = polar_encode(u_hat)
        hard_ch = (llr_ch < 0).astype(np.int8)
        return np.array_equal(x_hat, hard_ch)
