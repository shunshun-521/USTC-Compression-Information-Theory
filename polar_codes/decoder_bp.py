"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import build_generator_matrix, polar_encode


class BPDecoder:
    """BP 译码器（基于生成矩阵因子图的 min-sum BP）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

        G = build_generator_matrix(N)
        i_idx, j_idx = np.where(G)
        self.edge_i = i_idx.astype(np.int32)
        self.edge_j = j_idx.astype(np.int32)
        self.num_edges = len(self.edge_i)

        self.cn_edges = [[] for _ in range(N)]
        self.vn_edges = [[] for _ in range(N)]
        for eidx in range(self.num_edges):
            i = int(self.edge_i[eidx])
            j = int(self.edge_j[eidx])
            self.cn_edges[j].append(eidx)
            self.vn_edges[i].append(eidx)

        self._frozen_prior = np.zeros(N, dtype=np.float64)
        for i in range(N):
            if i in self.frozen_set:
                self._frozen_prior[i] = self.large

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        vn_to_cn = np.zeros(self.num_edges, dtype=np.float64)
        cn_to_vn = np.zeros(self.num_edges, dtype=np.float64)
        vn_total = np.zeros(self.N, dtype=np.float64)

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            np.add.at(vn_total, self.edge_i, cn_to_vn)
            for eidx in range(self.num_edges):
                i = int(self.edge_i[eidx])
                vn_to_cn[eidx] = self._frozen_prior[i] + vn_total[i] - cn_to_vn[eidx]

            for j in range(self.N):
                edge_ids = self.cn_edges[j]
                ne = len(edge_ids)
                if ne == 0:
                    continue
                ch = llr_ch[j]
                ch_sign = 1.0 if ch >= 0 else -1.0
                ch_abs = abs(ch)
                prod_sign = ch_sign
                min_abs = ch_abs
                for eidx in edge_ids:
                    m = vn_to_cn[eidx]
                    s = 1.0 if m >= 0 else -1.0
                    prod_sign *= s
                    a = abs(m)
                    if a < min_abs:
                        min_abs = a
                for k, eidx in enumerate(edge_ids):
                    m = vn_to_cn[eidx]
                    s = 1.0 if m >= 0 else -1.0
                    a = abs(m)
                    out_sign = prod_sign / s
                    if a > min_abs:
                        out_abs = min_abs
                    elif ne == 1:
                        out_abs = ch_abs
                    else:
                        out_abs = ch_abs
                        for e2 in edge_ids:
                            if e2 == eidx:
                                continue
                            a2 = abs(vn_to_cn[e2])
                            if a2 < out_abs:
                                out_abs = a2
                    cn_to_vn[eidx] = self.alpha * out_sign * out_abs

            vn_total.fill(0.0)
            u_hat = self._hard_decision(cn_to_vn)
            if self._early_stop(u_hat, llr_ch):
                num_iters = it
                break

        u_hat = self._hard_decision(cn_to_vn)
        return u_hat, num_iters

    def _hard_decision(self, cn_to_vn):
        total = self._frozen_prior.copy()
        np.add.at(total, self.edge_i, cn_to_vn)
        u_hat = (total < 0).astype(int)
        for idx in self.frozen_set:
            u_hat[idx] = 0
        return u_hat

    def _early_stop(self, u_hat, llr_ch):
        x_hat = polar_encode(u_hat)
        x_hard = (llr_ch < 0).astype(int)
        return np.array_equal(x_hat, x_hard)
