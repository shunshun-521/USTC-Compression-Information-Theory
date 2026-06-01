"""
极化码 BP（置信传播）译码器
基于校验矩阵的迭代加权比特翻转（WBF），含早停
"""
import numpy as np
from encoder import polar_encode


def _gf2_inv(A):
    n = A.shape[0]
    A = A.copy() % 2
    I = np.eye(n, dtype=int)
    for col in range(n):
        if A[col, col] == 0:
            sw = np.where(A[col:, col])[0]
            if len(sw) == 0:
                return None
            r = col + sw[0]
            A[[col, r]] = A[[r, col]]
            I[[col, r]] = I[[r, col]]
        for r in range(n):
            if r != col and A[r, col]:
                A[r] ^= A[col]
                I[r] ^= I[col]
    return I


def _get_Ginv(N):
    G = np.zeros((N, N), dtype=int)
    for i in range(N):
        u = np.zeros(N, dtype=int)
        u[i] = 1
        G[:, i] = polar_encode(u)
    return _gf2_inv(G)


class BPDecoder:
    """BP 风格迭代译码（校验域 WBF + 早停）"""

    _cache = {}

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha  # 保留接口，用于扩展 min-sum 因子
        if N not in BPDecoder._cache:
            BPDecoder._cache[N] = _get_Ginv(N)
        self.Ginv = BPDecoder._cache[N]
        frozen_pos = np.where(self.frozen_bits)[0]
        self.H = self.Ginv[frozen_pos, :]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        x = (llr_ch < 0).astype(int)
        u = (self.Ginv @ x) % 2
        u[self.frozen_bits] = 0

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            x_cur = polar_encode(u)
            syndrome = (self.H @ x_cur) % 2
            if not np.any(syndrome):
                num_iters = it
                break

            unsat = np.where(syndrome)[0]
            cand = set()
            for m in unsat:
                cand.update(np.where(self.H[m])[0].tolist())

            if not cand:
                break

            v = min(cand, key=lambda i: abs(llr_ch[i]))
            x_cur[v] ^= 1
            u = (self.Ginv @ x_cur) % 2
            u[self.frozen_bits] = 0
            num_iters = it

        x_hat = polar_encode(u)
        hard_ch = (llr_ch < 0).astype(int)
        if np.array_equal(x_hat, hard_ch) and num_iters == self.max_iter:
            num_iters = self.max_iter

        return u, num_iters
