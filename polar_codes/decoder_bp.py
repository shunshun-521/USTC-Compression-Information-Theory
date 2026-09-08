"""
极化码 BP（置信传播）译码器
基于校验矩阵 H 的 min-sum BP，含早停机制
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation, clip_llr

LARGE = 1e6


def _gf2_inverse(G):
    n = G.shape[0]
    A = G.copy().astype(np.int8)
    I = np.eye(n, dtype=np.int8)
    for col in range(n):
        if A[col, col] == 0:
            for row in range(col + 1, n):
                if A[row, col] == 1:
                    A[[col, row]] = A[[row, col]]
                    I[[col, row]] = I[[row, col]]
                    break
        for row in range(n):
            if row != col and A[row, col] == 1:
                A[row] ^= A[col]
                I[row] ^= I[col]
    return I


class BPDecoder:
    """BP 译码器（校验矩阵 min-sum BP）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

        enc = np.zeros((N, N), dtype=np.int8)
        for i in range(N):
            u = np.zeros(N, dtype=np.int8)
            u[i] = 1
            enc[i] = polar_encode(u)
        self.enc_matrix = enc
        enc_inv = _gf2_inverse(enc)
        frozen_idx = np.where(self.frozen_bits)[0]
        self.H = enc_inv[frozen_idx, :].astype(np.int8)
        self.enc_inv = enc_inv
        self.m = len(frozen_idx)

        self.vn_to_cn = [np.where(self.H[:, v] == 1)[0] for v in range(N)]
        self.cn_to_vn = [np.where(self.H[c, :] == 1)[0] for c in range(self.m)]

    def _minsum_f(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = clip_llr(np.asarray(llr_ch, dtype=np.float64))
        N, m = self.N, self.m

        R = np.zeros((m, N), dtype=np.float64)
        x_hat = np.zeros(N, dtype=np.int8)
        num_iters = 0

        for it in range(1, self.max_iter + 1):
            num_iters = it
            Q = np.zeros((m, N), dtype=np.float64)

            for v in range(N):
                for c in self.vn_to_cn[v]:
                    Q[c, v] = llr_ch[v] + np.sum(R[self.vn_to_cn[v], v]) - R[c, v]

            for c in range(m):
                vns = self.cn_to_vn[c]
                for v in vns:
                    others = [v2 for v2 in vns if v2 != v]
                    if not others:
                        R[c, v] = LARGE
                    else:
                        msg = Q[c, others[0]]
                        for v2 in others[1:]:
                            msg = self._minsum_f(msg, Q[c, v2])
                        R[c, v] = msg

            for v in range(N):
                total = llr_ch[v] + np.sum(R[self.vn_to_cn[v], v])
                x_hat[v] = 0 if total >= 0 else 1

            hard_ch = (llr_ch < 0).astype(np.int8)
            if np.array_equal(x_hat, hard_ch):
                break

        u_hat = np.mod(x_hat @ self.enc_inv, 2).astype(np.int8)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
