"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import build_generator_matrix, polar_encode


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """
    BP 译码器。
    在 u 节点与信道观测之间基于生成矩阵 G 的因子图进行 min-sum 消息传递。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6
        G = build_generator_matrix(N)
        self.edges = []
        for j in range(N):
            for i in np.where(G[:, j])[0]:
                self.edges.append((int(i), j))
        self.cn_edges = [[] for _ in range(N)]
        self.vn_edges = [[] for _ in range(N)]
        for eidx, (i, j) in enumerate(self.edges):
            self.cn_edges[j].append(eidx)
            self.vn_edges[i].append(eidx)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N

        vn_to_cn = np.zeros(len(self.edges), dtype=np.float64)
        cn_to_vn = np.zeros(len(self.edges), dtype=np.float64)

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            vn_prior = np.zeros(N, dtype=np.float64)
            for i in range(N):
                if i in self.frozen_set:
                    vn_prior[i] = self.large
                else:
                    vn_prior[i] = 0.0

            for eidx, (i, _) in enumerate(self.edges):
                total = vn_prior[i]
                for oe in self.vn_edges[i]:
                    if oe != eidx:
                        total += cn_to_vn[oe]
                vn_to_cn[eidx] = total

            for j in range(N):
                edge_ids = self.cn_edges[j]
                if not edge_ids:
                    continue
                msgs = np.array([vn_to_cn[e] for e in edge_ids], dtype=np.float64)
                all_msgs = np.concatenate(([llr_ch[j]], msgs))
                for k, eidx in enumerate(edge_ids):
                    others = np.delete(all_msgs, k + 1)
                    prod_sign = np.prod(np.sign(others + (others == 0)))
                    min_abs = np.min(np.abs(others))
                    cn_to_vn[eidx] = self.alpha * prod_sign * min_abs

            u_hat = np.zeros(N, dtype=int)
            for i in range(N):
                total = 0.0 if i not in self.frozen_set else self.large
                for eidx in self.vn_edges[i]:
                    total += cn_to_vn[eidx]
                u_hat[i] = 0 if total >= 0 else 1
            for idx in self.frozen_set:
                u_hat[idx] = 0

            if self._early_stop(u_hat, llr_ch):
                num_iters = it
                break

        u_hat = np.zeros(N, dtype=int)
        for i in range(N):
            total = 0.0 if i not in self.frozen_set else self.large
            for eidx in self.vn_edges[i]:
                total += cn_to_vn[eidx]
            u_hat[i] = 0 if total >= 0 else 1
        for idx in self.frozen_set:
            u_hat[idx] = 0

        return u_hat, num_iters

    def _early_stop(self, u_hat, llr_ch):
        x_hat = polar_encode(u_hat)
        x_hard = (llr_ch < 0).astype(int)
        return np.array_equal(x_hat, x_hard)
