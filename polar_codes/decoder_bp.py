"""
极化码 BP（置信传播）译码器
基于奇偶校验矩阵 H 的 min-sum BP，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode


def _gf2_inverse(mat):
    """GF(2) 矩阵求逆"""
    n = mat.shape[0]
    aug = np.concatenate([mat.copy(), np.eye(n, dtype=int)], axis=1)
    row = 0
    for col in range(n):
        pivot = None
        for r in range(row, n):
            if aug[r, col]:
                pivot = r
                break
        if pivot is None:
            raise ValueError("Matrix not invertible over GF(2)")
        if pivot != row:
            aug[[row, pivot]] = aug[[pivot, row]]
        for r in range(n):
            if r != row and aug[r, col]:
                aug[r] ^= aug[row]
        row += 1
    return aug[:, n:].astype(int)


def _build_generator_matrix(N):
    """从编码器构造 G，满足 x = u @ G (mod 2)"""
    G = np.zeros((N, N), dtype=int)
    for j in range(N):
        u = np.zeros(N, dtype=int)
        u[j] = 1
        G[j, :] = polar_encode(u)
    return G


def _parity_matrix(N, frozen_bits):
    """由冻结位约束构造奇偶校验矩阵 H"""
    G = _build_generator_matrix(N)
    Ginv = _gf2_inverse(G)
    frozen_idx = np.where(np.asarray(frozen_bits, dtype=int) == 1)[0]
    return Ginv[:, frozen_idx].T


def _g_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（min-sum，基于 H 矩阵）"""

    LARGE = 1e7

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.H = _parity_matrix(N, frozen_bits)
        self.Ginv = _gf2_inverse(_build_generator_matrix(N))
        self._checks = [np.where(self.H[m])[0] for m in range(self.H.shape[0])]
        self._vars = [np.where(self.H[:, n])[0] for n in range(N)]

    def _hard_llr(self, llr_ch):
        return (llr_ch < 0).astype(int)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        M = self.H.shape[0]

        channel = llr_ch.copy()

        Lq = channel.copy()
        Rmn = np.zeros((M, N), dtype=np.float64)

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for m in range(M):
                vars_m = self._checks[m]
                for n in vars_m:
                    sign_prod = 1.0
                    min_val = np.inf
                    for n2 in vars_m:
                        if n2 == n:
                            continue
                        val = Lq[n2] - Rmn[m, n2]
                        if val == 0:
                            continue
                        sign_prod *= np.sign(val)
                        min_val = min(min_val, abs(val))
                    if min_val == np.inf:
                        Rmn[m, n] = 0.0
                    else:
                        Rmn[m, n] = self.alpha * sign_prod * min_val

            for n in range(N):
                Lq[n] = channel[n]
                for m in self._vars[n]:
                    Lq[n] += Rmn[m, n]

            u_tmp = np.dot(x_hat, self.Ginv) % 2
            u_tmp[self.frozen_bits == 1] = 0
            if np.array_equal(polar_encode(u_tmp), self._hard_llr(llr_ch)):
                num_iters = it
                break

        x_hat = (Lq < 0).astype(int)
        u_hat = np.dot(x_hat, self.Ginv) % 2
        u_hat = u_hat.astype(int)
        u_hat[self.frozen_bits == 1] = 0

        return u_hat, num_iters
