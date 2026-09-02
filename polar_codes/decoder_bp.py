"""
极化码 BP（置信传播）译码器
基于等效 Tanner 图（校验矩阵 H = G^{-1}）的 min-sum BP，含早停机制
"""
import numpy as np
from decoder_sc import f_operation
from encoder import build_generator_matrix, polar_encode


def _gf2_inverse(matrix):
    """GF(2) 矩阵求逆。"""
    mat = np.asarray(matrix, dtype=np.int8).copy() % 2
    n = mat.shape[0]
    aug = np.concatenate([mat, np.eye(n, dtype=np.int8)], axis=1)
    for col in range(n):
        pivot = next(row for row in range(col, n) if aug[row, col] == 1)
        if pivot != col:
            aug[[col, pivot]] = aug[[pivot, col]]
        for row in range(n):
            if row != col and aug[row, col]:
                aug[row] ^= aug[col]
    return aug[:, n:]


class BPDecoder:
    """BP 译码器（min-sum，基于极化码校验矩阵）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=np.int8)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]
        G = build_generator_matrix(N)
        self.H = _gf2_inverse(G)

    def _f_min_sum(self, msgs):
        if not msgs:
            return 0.0
        sign = np.prod(np.sign(msgs))
        return self.alpha * sign * np.min(np.abs(msgs))

    def _bp_codeword(self, llr_ch):
        """在码字比特上执行 min-sum BP。"""
        m, n = self.H.shape
        Lq = llr_ch.astype(np.float64).copy()
        Rcv = np.zeros((m, n), dtype=np.float64)

        for _ in range(self.max_iter):
            for i in range(m):
                idx = np.where(self.H[i] == 1)[0]
                for j in idx:
                    msgs = [Lq[k] + Rcv[i, k] for k in idx if k != j]
                    Rcv[i, j] = self._f_min_sum(msgs)
        return Lq + Rcv.sum(axis=0)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        l_total = self._bp_codeword(llr_ch)
        x_hat = (l_total < 0).astype(np.int8)
        u_hat = (x_hat @ self.H) % 2
        u_hat[self.frozen_idx] = 0
        return u_hat, self.max_iter
