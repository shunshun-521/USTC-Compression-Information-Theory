"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode


class BPDecoder:
    """BP 译码器（参考分层消息传递实现）。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def _f_min_sum(self, a, b):
        sa = 1.0 if np.sign(a) == 0 else np.sign(a)
        sb = 1.0 if np.sign(b) == 0 else np.sign(b)
        return self.alpha * sa * sb * min(abs(a), abs(b))

    def _update_left(self, left_col, right_col, stage):
        """左向（信道→信源）L 消息更新。"""
        N = self.N
        interval = 1 << (stage - 1)
        num = N // (interval * 2)
        out = np.zeros(N, dtype=np.float64)
        for block in range(num):
            base = 2 * block * interval
            for j in range(interval):
                i0 = base + j
                i1 = base + j + interval
                left = np.array([left_col[i0], left_col[i1]])
                right = np.array([right_col[i0], right_col[i1]])
                out[i0] = self._f_min_sum(right[1] + left[1], left[0])
                out[i1] = self._f_min_sum(left[0], right[0]) + left[1]
        return out

    def _update_right(self, left_col, right_col, stage):
        """右向（信源→信道）R 消息更新。"""
        N = self.N
        interval = 1 << (stage - 1)
        num = N // (interval * 2)
        out = np.zeros(N, dtype=np.float64)
        for block in range(num):
            base = 2 * block * interval
            for j in range(interval):
                i0 = base + j
                i1 = base + j + interval
                left = np.array([left_col[i0], left_col[i1]])
                right = np.array([right_col[i0], right_col[i1]])
                out[i0] = self._f_min_sum(right[1] + left[1], right[0])
                out[i1] = self._f_min_sum(left[0], right[0]) + right[1]
        return out

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

        u_hat = np.zeros(N, dtype=np.int8)
        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for i in range(n):
                L[:, n - i - 1] = self._update_left(L[:, n - i], R[:, n - i - 1], n - i)

            for i in range(n):
                R[:, i + 1] = self._update_right(L[:, i + 1], R[:, i], i + 1)

            total = L[:, 0] + R[:, 0]
            for idx in range(N):
                if self.frozen_bits[idx]:
                    u_hat[idx] = 0
                else:
                    u_hat[idx] = 0 if total[idx] >= 0 else 1

            x_hat = polar_encode(u_hat)
            x_llr = L[:, n] + R[:, n]
            x_hard = (x_llr < 0).astype(np.int8)
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        for idx in range(N):
            if self.frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if total[idx] >= 0 else 1

        return u_hat, num_iters
