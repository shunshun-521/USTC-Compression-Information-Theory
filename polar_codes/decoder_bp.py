"""
极化码 BP（置信传播）译码器
基于极化码校验矩阵的因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import build_generator_matrix, polar_encode


class BPDecoder:
    """BP 译码器（校验矩阵 H = G^T[frozen] 的 Tanner 图）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

        G = build_generator_matrix(N)
        frozen_idx = np.where(self.frozen_bits)[0]
        self.H = G.T[frozen_idx].astype(np.int8)
        self.n_checks = self.H.shape[0]

        self.edges = []
        self.chk_edges = [[] for _ in range(self.n_checks)]
        self.var_edges = [[] for _ in range(N)]
        eid = 0
        for c in range(self.n_checks):
            for v in np.where(self.H[c])[0]:
                self.edges.append((c, int(v)))
                self.chk_edges[c].append(eid)
                self.var_edges[int(v)].append(eid)
                eid += 1
        self.n_edges = eid

    def _cn_update(self, incoming):
        if len(incoming) == 1:
            return np.array([self.alpha * incoming[0]])
        signs = np.prod(np.sign(incoming + 1e-12))
        abs_vals = np.abs(incoming)
        out = np.empty(len(incoming), dtype=np.float64)
        for i in range(len(incoming)):
            others = np.delete(abs_vals, i)
            out[i] = (
                self.alpha
                * signs
                * np.sign(incoming[i] + 1e-12)
                * np.min(others)
            )
        return out

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N

        var_to_chk = np.zeros(self.n_edges, dtype=np.float64)
        chk_to_var = np.zeros(self.n_edges, dtype=np.float64)

        for eid, (_, v) in enumerate(self.edges):
            var_to_chk[eid] = llr_ch[v]

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for c in range(self.n_checks):
                edge_ids = self.chk_edges[c]
                msgs = np.array([var_to_chk[e] for e in edge_ids])
                out = self._cn_update(msgs)
                for e, val in zip(edge_ids, out):
                    chk_to_var[e] = val

            total_llr = llr_ch.copy()
            for v in range(N):
                total_llr[v] += sum(chk_to_var[e] for e in self.var_edges[v])

            for eid, (c, v) in enumerate(self.edges):
                other = sum(chk_to_var[e] for e in self.var_edges[v] if e != eid)
                var_to_chk[eid] = llr_ch[v] + other

            for v in range(N):
                if self.frozen_bits[v]:
                    u_hat[v] = 0
                else:
                    u_hat[v] = 0 if total_llr[v] >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        for v in range(N):
            total = llr_ch[v] + sum(chk_to_var[e] for e in self.var_edges[v])
            u_hat[v] = 0 if self.frozen_bits[v] or total >= 0 else 1

        return u_hat, num_iters
