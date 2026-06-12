"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _ms_f(x, y, alpha):
    s1 = np.sign(x)
    s2 = np.sign(y)
    s1 = np.where(s1 == 0, 1, s1)
    s2 = np.where(s2 == 0, 1, s2)
    return alpha * s1 * s2 * np.minimum(np.abs(x), np.abs(y))


def _bp_update_left(left_col, right_col, stage, alpha):
    """左向（信道 -> 信源）消息更新。"""
    N = len(left_col)
    interval = 1 << (stage - 1)
    out = np.zeros(N, dtype=np.float64)
    num = N // (2 * interval)
    for i in range(num):
        for j in range(interval):
            base = 2 * i * interval + j
            left_ele = np.array([left_col[base], left_col[base + interval]])
            right_ele = np.array([right_col[base], right_col[base + interval]])
            out[base] = _ms_f(right_ele[1] + left_ele[1], left_ele[0], alpha)
            out[base + interval] = _ms_f(left_ele[0], right_ele[0], alpha) + left_ele[1]
    return out


def _bp_update_right(left_col, right_col, stage, alpha):
    """右向（信源 -> 信道）消息更新。"""
    N = len(left_col)
    interval = 1 << (stage - 1)
    out = np.zeros(N, dtype=np.float64)
    num = N // (2 * interval)
    for i in range(num):
        for j in range(interval):
            base = 2 * i * interval + j
            left_ele = np.array([left_col[base], left_col[base + interval]])
            right_ele = np.array([right_col[base], right_col[base + interval]])
            out[base] = _ms_f(right_ele[1] + left_ele[1], right_ele[0], alpha)
            out[base + interval] = _ms_f(left_ele[0], right_ele[0], alpha) + right_ele[1]
    return out


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        assert 2**self.n == N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, num_iters
        """
        llr_raw = np.asarray(llr_ch, dtype=np.float64)
        br = bit_reversal_permutation(self.N)
        llr_ch = llr_raw[br]
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for i in range(n):
                stage = n - i
                L[:, stage - 1] = _bp_update_left(
                    L[:, stage], R[:, stage - 1], stage, self.alpha
                )

            for i in range(n):
                stage = i + 1
                R[:, stage] = _bp_update_right(
                    L[:, stage], R[:, stage - 1], stage, self.alpha
                )

            posterior = L[:, 0] + R[:, 0]
            for idx in range(N):
                u_hat[idx] = 0 if posterior[idx] >= 0 else 1
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard = (llr_raw < 0).astype(int)
            if np.array_equal(x_hat, hard):
                num_iters = it
                break

        posterior = L[:, 0] + R[:, 0]
        for idx in range(N):
            u_hat[idx] = 0 if posterior[idx] >= 0 else 1
        u_hat[self.frozen_bits] = 0

        return u_hat, num_iters
