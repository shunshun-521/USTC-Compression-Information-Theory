"""
极化码 BP（置信传播）译码器
在极化码校验矩阵 H 上运行 min-sum BP，含早停
"""
import numpy as np
from encoder import polar_encode


def _build_G(N):
    F = np.array([[1, 0], [1, 1]], dtype=int)
    n = int(np.log2(N))
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    return G


def _gf2_inverse(A):
    n = A.shape[0]
    aug = np.concatenate([A.astype(int) % 2, np.eye(n, dtype=int)], axis=1)
    row = 0
    for col in range(n):
        pivot = next((r for r in range(row, n) if aug[r, col]), None)
        if pivot is None:
            raise ValueError("Matrix not invertible")
        if pivot != row:
            aug[[row, pivot]] = aug[[pivot, row]]
        for r in range(n):
            if r != row and aug[r, col]:
                aug[r] ^= aug[row]
        row += 1
    return aug[:, n:] % 2


def _cn_minsum_vec(msgs, alpha):
    """向量化的校验节点 min-sum。"""
    msgs = np.asarray(msgs, dtype=np.float64)
    if msgs.size == 0:
        return 0.0
    if msgs.size == 1:
        return float(msgs[0])
    signs = np.sign(msgs)
    signs[signs == 0] = 1.0
    return float(alpha * np.prod(signs) * np.min(np.abs(msgs)))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

        G_inv = _gf2_inverse(_build_G(N))
        frozen_idx = np.where(self.frozen_bits)[0]
        self.G_inv = G_inv
        self.H = G_inv[:, frozen_idx].T.astype(int)
        M = self.H.shape[0]
        self.check_to_var = [np.where(self.H[c])[0] for c in range(M)]
        self.var_to_check = [np.where(self.H[:, i])[0] for i in range(N)]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        alpha = self.alpha
        M = self.H.shape[0]

        Lv = llr_ch.copy()
        Lvc = np.zeros((M, N), dtype=np.float64)
        Lcv = np.zeros((M, N), dtype=np.float64)

        num_iters = self.max_iter
        ctv = self.check_to_var
        vtc = self.var_to_check

        for it in range(1, self.max_iter + 1):
            for i in range(N):
                for c in vtc[i]:
                    Lcv[c, i] = Lv[i] - Lvc[c, i]

            for c in range(M):
                nodes = ctv[c]
                for idx_i, i in enumerate(nodes):
                    msgs = [Lcv[c, j] for j in nodes if j != i]
                    Lvc[c, i] = _cn_minsum_vec(msgs, alpha)

            for i in range(N):
                cs = vtc[i]
                if cs.size:
                    Lv[i] = llr_ch[i] + Lvc[cs, i].sum()
                else:
                    Lv[i] = llr_ch[i]

            x_hat = (Lv < 0).astype(int)
            if np.array_equal(x_hat, (llr_ch < 0).astype(int)):
                num_iters = it
                break

        x_hat = (Lv < 0).astype(int)
        u_hat = np.mod(x_hat.dot(self.G_inv), 2).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
