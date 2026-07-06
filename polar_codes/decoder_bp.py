"""
极化码 BP（置信传播）译码器
基于生成矩阵因子图的 min-sum BP，含早停机制
"""
import numpy as np

from encoder import polar_encode


class BPDecoder:
    """
    BP 译码器：在 x = u @ G 的因子图上进行置信传播。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

        G = np.zeros((N, N), dtype=np.int8)
        for i in range(N):
            e = np.zeros(N, dtype=np.int8)
            e[i] = 1
            G[i] = polar_encode(e)

        var_idx, chk_idx = np.where(G)
        self.var_idx = var_idx.astype(np.int32)
        self.chk_idx = chk_idx.astype(np.int32)
        self.n_edges = len(self.var_idx)

        self.var_edges = [np.where(self.var_idx == i)[0] for i in range(N)]
        self.chk_edges = [np.where(self.chk_idx == j)[0] for j in range(N)]

        self._chk_edge_vars = [self.var_idx[e] for e in range(self.n_edges)]

    def _f_minsum_scalar(self, x, y):
        return self.alpha * np.sign(x) * np.sign(y) * min(abs(x), abs(y))

    def _boxplus_list(self, msgs):
        result = msgs[0]
        for m in msgs[1:]:
            result = self._f_minsum_scalar(result, m)
        return result

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat 和实际迭代次数。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n_edges = self.n_edges
        frozen = self.frozen_bits
        var_edges = self.var_edges
        chk_edges = self.chk_edges
        chk_edge_vars = self._chk_edge_vars
        large = self.large

        v2c = np.zeros(n_edges, dtype=np.float64)
        c2v = np.zeros(n_edges, dtype=np.float64)

        for i in range(N):
            if frozen[i]:
                v2c[var_edges[i]] = large

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(N):
                edges = chk_edges[j]
                ch_llr = llr_ch[j]
                for e in edges:
                    others = [v2c[oe] for oe in edges if oe != e]
                    c2v[e] = self._boxplus_list([ch_llr] + others)

            for i in range(N):
                edges = var_edges[i]
                if frozen[i]:
                    v2c[edges] = large
                    continue
                for e in edges:
                    others = [c2v[oe] for oe in edges if oe != e]
                    v2c[e] = np.sum(others) if others else 0.0

            for i in range(N):
                if frozen[i]:
                    u_hat[i] = 0
                else:
                    total = np.sum(c2v[var_edges[i]])
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, (llr_ch < 0).astype(int)):
                num_iters = it
                break

        for i in range(N):
            if frozen[i]:
                u_hat[i] = 0
            else:
                total = np.sum(c2v[var_edges[i]])
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat, num_iters
