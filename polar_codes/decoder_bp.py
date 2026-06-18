"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import polar_encode


def _f_min_sum(L1, L2, alpha=0.9375):
    s1 = 1 if L1 == 0 else np.sign(L1)
    s2 = 1 if L2 == 0 else np.sign(L2)
    return alpha * s1 * s2 * min(abs(L1), abs(L2))


def _pe_update_left(left_col, right_col, stage):
    N = left_col.size
    interval = 2 ** (stage - 1)
    out = np.zeros(N, dtype=np.float64)
    num = N // (interval * 2)
    for block in range(num):
        base = 2 * block * interval
        for j in range(interval):
            i0 = base + j
            i1 = base + j + interval
            l0, l1 = left_col[i0], left_col[i1]
            r0, r1 = right_col[i0], right_col[i1]
            out[i0] = _f_min_sum(r1 + l1, l0)
            out[i1] = _f_min_sum(l0, r0) + l1
    return out


def _pe_update_right(left_col, right_col, stage):
    N = left_col.size
    interval = 2 ** (stage - 1)
    out = np.zeros(N, dtype=np.float64)
    num = N // (interval * 2)
    for block in range(num):
        base = 2 * block * interval
        for j in range(interval):
            i0 = base + j
            i1 = base + j + interval
            l0, l1 = left_col[i0], left_col[i1]
            r0, r1 = right_col[i0], right_col[i1]
            out[i0] = _f_min_sum(r1 + l1, r0)
            out[i1] = _f_min_sum(l0, r0) + r1
    return out


class BPDecoder:
    """
    BP 译码器。
    因子图有 n+1 列（列 0 到列 n），每列 N 个节点。
    """

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, num_iters
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for i in range(n):
                stage = n - i
                L[:, n - i - 1] = _pe_update_left(L[:, n - i], R[:, n - i - 1], stage)

            for i in range(n):
                R[:, i + 1] = _pe_update_right(L[:, i + 1], R[:, i], i + 1)

            u_hat = self._hard_decision(L, R)
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        u_hat = self._hard_decision(L, R)
        return u_hat, num_iters

    def _hard_decision(self, L, R):
        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat
