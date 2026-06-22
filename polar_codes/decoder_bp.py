"""
极化码 BP（置信传播）译码器
基于因子图/校验矩阵，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _build_generator_matrix(N):
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    B = np.zeros((N, N), dtype=np.int8)
    br = bit_reversal_permutation(N)
    for i, j in enumerate(br):
        B[i, j] = 1
    return (B @ G) % 2


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

        G = _build_generator_matrix(N)
        frozen_idx = np.where(self.frozen_bits)[0]
        self.H = G[:, frozen_idx].T.astype(np.int8)
        self.G = G

        M = self.H.shape[0]
        self._cn_neighbors = [np.where(self.H[m])[0] for m in range(M)]
        self._vn_neighbors = [np.where(self.H[:, n])[0] for n in range(N)]

    def _min_sum_check_update(self, Lq, Lr):
        for m, nbr in enumerate(self._cn_neighbors):
            msgs = [Lq[j] - Lr[m, j] for j in nbr]
            for idx, j in enumerate(nbr):
                others = [msgs[k] for k in range(len(nbr)) if k != idx]
                if not others:
                    Lr[m, j] = 0.0
                    continue
                prod_sign = np.prod(np.sign(others))
                min_abs = min(abs(v) for v in others)
                Lr[m, j] = self.alpha * prod_sign * min_abs

    def _variable_update(self, llr_ch, Lq, Lr):
        for j in range(self.N):
            ext = sum(Lr[m, j] for m in self._vn_neighbors[j])
            Lq[j] = llr_ch[j] + ext

    def _codeword_to_source(self, x_hat):
        u_hat = np.mod(x_hat @ self.G, 2).astype(np.int8)
        u_hat[self.frozen_bits] = 0
        return u_hat

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        Lq = llr_ch.copy()
        Lr = np.zeros_like(self.H, dtype=np.float64)
        num_iters = self.max_iter
        u_hat = np.zeros(self.N, dtype=np.int8)

        for it in range(1, self.max_iter + 1):
            self._min_sum_check_update(Lq, Lr)
            self._variable_update(llr_ch, Lq, Lr)

            x_hat = (Lq < 0).astype(np.int8)
            u_hat = self._codeword_to_source(x_hat)

            x_reenc = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(np.int8)
            if np.array_equal(x_reenc, hard_ch):
                num_iters = it
                break

        x_hat = (Lq < 0).astype(np.int8)
        u_hat = self._codeword_to_source(x_hat)
        return u_hat.astype(int), num_iters
