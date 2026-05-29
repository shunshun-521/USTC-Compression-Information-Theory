"""
极化码 BP（置信传播）译码器
基于冻结位导出的校验矩阵 H 进行标准 min-sum BP
"""
import numpy as np
import math
from encoder import polar_encode, generator_matrix


class BPDecoder:
    """BP 译码器（LDPC min-sum on polar parity matrix）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        if 2**self.n != N:
            raise ValueError(f"N={N} must be a power of 2")
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self._large = 20.0

        self.G = generator_matrix(N)
        self.H = self.G[:, self.frozen_idx].T.astype(np.int8)
        self.M = len(self.frozen_idx)

        self._cn_neighbors = [np.where(self.H[m])[0] for m in range(self.M)]
        self._vn_neighbors = [np.where(self.H[:, v])[0] for v in range(N)]

    def decode(self, llr_ch):
        llr_ch = np.clip(np.asarray(llr_ch, dtype=np.float64), -self._large, self._large)
        N = self.N
        M = self.M

        r_cv = np.zeros((M, N), dtype=np.float64)
        q_vc = np.zeros((M, N), dtype=np.float64)

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=np.int8)

        for it in range(1, self.max_iter + 1):
            # VN -> CN
            for v in range(N):
                for m in self._vn_neighbors[v]:
                    q_vc[m, v] = llr_ch[v] + sum(
                        r_cv[mp, v] for mp in self._vn_neighbors[v] if mp != m
                    )
                    q_vc[m, v] = np.clip(q_vc[m, v], -self._large, self._large)

            # CN -> VN (min-sum)
            for m in range(M):
                nbrs = self._cn_neighbors[m]
                for v in nbrs:
                    sign = 1.0
                    mag = self._large
                    for o in nbrs:
                        if o == v:
                            continue
                        val = q_vc[m, o]
                        sign *= np.sign(val) if val != 0 else 1.0
                        mag = min(mag, abs(val))
                    r_cv[m, v] = self.alpha * sign * mag

            total = np.zeros(N, dtype=np.float64)
            for v in range(N):
                total[v] = llr_ch[v] + sum(r_cv[m, v] for m in self._vn_neighbors[v])
            x_hat = (total < 0).astype(np.int8)
            u_hat = (x_hat @ self.G) % 2
            u_hat[self.frozen_bits] = 0

            if self._early_stop(u_hat, llr_ch):
                num_iters = it
                break

        total = np.zeros(N, dtype=np.float64)
        for v in range(N):
            total[v] = llr_ch[v] + sum(r_cv[m, v] for m in self._vn_neighbors[v])
        x_hat = (total < 0).astype(np.int8)
        u_hat = (x_hat @ self.G) % 2
        u_hat[self.frozen_bits] = 0
        return u_hat.astype(np.int8), num_iters

    def _early_stop(self, u_hat, llr_ch):
        x_hat = polar_encode(u_hat)
        hard_ch = (llr_ch < 0).astype(np.int8)
        return np.array_equal(x_hat, hard_ch)
