"""
极化码 BP（置信传播）译码器
基于校验矩阵的 min-sum BP，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode


def _build_generator(N):
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    while G.shape[0] < N:
        G = np.kron(G, F)
    n = int(math.log2(N))
    B = np.zeros((N, N), dtype=int)
    for i in range(N):
        B[int(format(i, f"0{n}b")[::-1], 2), i] = 1
    return (B @ G) % 2


def _gf2_inverse(matrix):
    A = matrix.copy() % 2
    n = A.shape[0]
    aug = np.concatenate([A, np.eye(n, dtype=int)], axis=1)
    row = 0
    for col in range(n):
        pivot = next(i for i in range(row, n) if aug[i, col])
        aug[[row, pivot]] = aug[[pivot, row]]
        for i in range(n):
            if i != row and aug[i, col]:
                aug[i] ^= aug[row]
        row += 1
    return aug[:, n:] % 2


class BPDecoder:
    """BP 译码器（在校验矩阵 Tanner 图上执行 min-sum BP）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.info_idx = np.where(~self.frozen_bits)[0]

        G = _build_generator(N)
        U = _gf2_inverse(G)
        self._U = U
        self._H = U[:, self.frozen_idx].T % 2

        self._cn = [np.where(self._H[m])[0] for m in range(self._H.shape[0])]

    def _bp_decode_codeword(self, llr):
        M, N = self._H.shape
        var_to_check = np.zeros((M, N), dtype=np.float64)
        check_to_var = np.zeros((M, N), dtype=np.float64)
        for m, cols in enumerate(self._cn):
            for v in cols:
                var_to_check[m, v] = llr[v]

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for m, cols in enumerate(self._cn):
                incoming = var_to_check[m, cols] - check_to_var[m, cols]
                for k, v in enumerate(cols):
                    others = np.delete(incoming, k)
                    if len(others) == 0:
                        check_to_var[m, v] = 0.0
                    else:
                        sign = np.prod(np.sign(others))
                        mag = np.min(np.abs(others))
                        check_to_var[m, v] = self.alpha * sign * mag

            posterior = llr + np.sum(check_to_var, axis=0)
            hard = (posterior < 0).astype(int)
            if np.all((self._H @ hard) % 2 == 0):
                num_iters = it
                return hard, num_iters

        posterior = llr + np.sum(check_to_var, axis=0)
        return (posterior < 0).astype(int), num_iters

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, num_iters
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        codeword_hat, num_iters = self._bp_decode_codeword(llr_ch)
        u_hat = (codeword_hat @ self._U) % 2
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
