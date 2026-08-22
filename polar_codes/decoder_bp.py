"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from decoder_sc import f_operation
from encoder import polar_encode


def _index_matrix(N):
    """构造极化码 BP 因子图各阶段的节点掩码。"""
    x = np.arange(1, N + 1)
    n = int(math.log2(N))
    M = np.zeros((N - 1, n), dtype=np.int32)
    for k in range(n):
        step = 2 ** (k + 1)
        half = 2 ** k
        for i in range(0, N, step):
            if i + half < N:
                M[i : i + half, n - k - 1] = x[i : i + half]
    mat = M.T[M.T > 0].reshape(n, N // 2).T - 1
    return mat[np.flip(np.arange(n))]


def _butterfly_codeword(u):
    """蝶形编码（不含比特倒序），用于早停校验。"""
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(math.log2(N))
    for layer in range(n):
        step = 1 << (layer + 1)
        half = step >> 1
        for i in range(0, N, step):
            for j in range(i, i + half):
                u[j] ^= u[j + half]
    return u


class BPDecoder:
    """
    BP 译码器（参考 Kaira/Arikan 因子图结构）。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.mask_dict = _index_matrix(N)
        self.perm = np.arange(self.n)

    def _checknode(self, y1, y2):
        return self.alpha * f_operation(y1, y2)

    def _update_left(self, R, L):
        m, N = self.n, self.N
        for i in np.flip(self.perm):
            i_back = m - i - 1
            add_k = N // (2 ** (i_back + 1))
            mask = self.mask_dict[i]
            if len(mask) == 0:
                continue
            idx = mask
            idx2 = mask + add_k
            L[i, idx] = self._checknode(
                L[i + 1, idx], L[i + 1, idx2] + R[i, idx2]
            )
            L[i, idx2] = self._checknode(R[i, idx], L[i + 1, idx]) + L[i + 1, idx2]
        return L

    def _update_right(self, R, L):
        m, N = self.n, self.N
        for i in self.perm:
            i_back = m - i - 1
            add_k = N // (2 ** (i_back + 1))
            mask = self.mask_dict[i]
            if len(mask) == 0:
                continue
            idx = mask
            idx2 = mask + add_k
            R[i + 1, idx] = self._checknode(
                R[i, idx], L[i + 1, idx2] + R[i, idx2]
            )
            R[i + 1, idx2] = self._checknode(R[i, idx], L[i + 1, idx]) + R[i, idx2]
        return R

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：
            u_hat: 长度 N 的估计源序列
            num_iters: 实际迭代次数
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        m = self.n
        LARGE = 1e6

        R = np.zeros((m + 1, self.N), dtype=np.float64)
        L = np.zeros((m + 1, self.N), dtype=np.float64)
        R[0, self.frozen_idx] = LARGE
        L[m, :] = llr_ch

        num_iters = 0
        u_hat = np.zeros(self.N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it
            L = self._update_left(R, L)
            R = self._update_right(R, L)

            for i in range(self.N):
                total = L[0, i] + R[0, i]
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total >= 0 else 1

            v_hat = _butterfly_codeword(u_hat)
            hard_v = (llr_ch < 0).astype(int)
            if np.array_equal(v_hat, hard_v):
                break

        for i in range(self.N):
            total = L[0, i] + R[0, i]
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat, num_iters
