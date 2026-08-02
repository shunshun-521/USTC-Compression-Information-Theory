"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode


def _build_G(N):
    G = np.array([[1]])
    while G.shape[0] < N:
        Z = np.zeros_like(G)
        G = np.block([[G, Z], [G, G]])
    return G.astype(int) % 2


class BPDecoder:
    """
    BP 译码器（基于极化码校验矩阵 H=G 的因子图）。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self._large = 1e6
        self.H = _build_G(N)

    def _minsum(self, a, b):
        sa, sb = np.sign(a), np.sign(b)
        return self.alpha * sa * sb * np.minimum(np.abs(a), np.abs(b))

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, num_iters)
        """
        N = self.N
        H = self.H
        M = N
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        var_to_check = np.tile(llr_ch, (M, 1))
        check_to_var = np.zeros((M, N))

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it

            for c in range(M):
                idx = np.where(H[c])[0]
                if len(idx) == 0:
                    continue
                msgs = var_to_check[c, idx] + check_to_var[c, idx]
                for k, v in enumerate(idx):
                    others = [msgs[j] for j in range(len(idx)) if j != k]
                    if len(others) == 0:
                        prod = 0.0
                    elif len(others) == 1:
                        prod = others[0]
                    else:
                        prod = self._reduce(others)
                    check_to_var[c, v] = prod

            for v in range(N):
                checks = np.where(H[:, v])[0]
                total = llr_ch[v] + np.sum(check_to_var[checks, v])
                for c in checks:
                    var_to_check[c, v] = total - check_to_var[c, v]

            total_llr = llr_ch + np.sum(check_to_var, axis=0)
            u_hat = np.where(total_llr >= 0, 0, 1).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        return u_hat, num_iters

    def _reduce(self, msgs):
        result = msgs[0]
        for m in msgs[1:]:
            result = self._minsum(result, m)
        return result
