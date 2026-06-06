"""
极化码 BP（置信传播）译码器
基于 Tanner 图的 min-sum BP，含早停机制
"""
import numpy as np

from encoder import polar_encode


def _build_generator_matrix(N):
    """构造 G_N = F^⊗n。"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F
    n = int(np.log2(N))
    for _ in range(n - 1):
        G = np.kron(G, F)
    return G


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e8

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.G = _build_generator_matrix(N)

        self.var_to_check = [[] for _ in range(N)]
        self.check_to_var = [[] for _ in range(N)]
        for i in range(N):
            for j in range(N):
                if self.G[i, j]:
                    self.var_to_check[i].append(j)
                    self.check_to_var[j].append(i)

    @staticmethod
    def _f_min_sum(a, b, alpha):
        return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))

    def _parity_to_var(self, check_idx, target_var, Lq, llr_ch):
        """校验节点到变量节点的 min-sum 消息。"""
        others = [v for v in self.check_to_var[check_idx] if v != target_var]
        msg = llr_ch[check_idx]
        for v in others:
            msg = self._f_min_sum(msg, Lq[v, check_idx], self.alpha)
        return msg

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, num_iters)。"""
        N = self.N
        Lq = np.zeros((N, N), dtype=np.float64)
        Lr = np.zeros((N, N), dtype=np.float64)

        prior = np.zeros(N, dtype=np.float64)
        prior[self.frozen_bits] = self.LARGE

        num_iters = self.max_iter
        LQ_total = prior.copy()

        for it in range(1, self.max_iter + 1):
            for j in range(N):
                for i in self.check_to_var[j]:
                    Lr[j, i] = self._parity_to_var(j, i, Lq, llr_ch)

            for i in range(N):
                total = prior[i]
                for j in self.var_to_check[i]:
                    total += Lr[j, i]
                LQ_total[i] = total
                for j in self.var_to_check[i]:
                    Lq[i, j] = total - Lr[j, i]

            u_hat = np.zeros(N, dtype=int)
            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if LQ_total[i] >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        u_hat = np.zeros(N, dtype=int)
        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if LQ_total[i] >= 0 else 1

        return u_hat, num_iters
