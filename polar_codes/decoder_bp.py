"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode


def _sign_pm(x):
    """min-sum 符号：0 视为 +1"""
    return np.where(x >= 0, 1.0, -1.0)


def _f_min_sum(x, y, alpha):
    """min-sum f 运算，带修正因子 alpha；单边为 0 时传递另一边"""
    ax = np.abs(x)
    ay = np.abs(y)
    zero_x = ax < 1e-15
    zero_y = ay < 1e-15
    both = ~(zero_x | zero_y)
    out = np.zeros_like(x, dtype=np.float64)
    out[both] = (
        alpha
        * _sign_pm(x[both])
        * _sign_pm(y[both])
        * np.minimum(ax[both], ay[both])
    )
    out[zero_x & ~zero_y] = y[zero_x & ~zero_y]
    out[zero_y & ~zero_x] = x[zero_y & ~zero_x]
    return out


def _element_update_left(left_top, left_bot, right_top, right_bot, alpha):
    """左向 L 消息更新（2 元 PE）"""
    out_top = _f_min_sum(right_bot + left_bot, left_top, alpha)
    out_bot = _f_min_sum(left_top, right_top, alpha) + left_bot
    return out_top, out_bot


def _element_update_right(left_top, left_bot, right_top, right_bot, alpha):
    """右向 R 消息更新（2 元 PE）"""
    out_top = _f_min_sum(right_bot + left_bot, right_top, alpha)
    out_bot = _f_min_sum(left_top, right_top, alpha) + right_bot
    return out_top, out_bot


def _bp_update_left(L_right, R_left, layer, alpha):
    """整列左向更新，layer 为 1..n"""
    N = len(L_right)
    interval = 1 << (layer - 1)
    num = N // (2 * interval)
    out = np.zeros(N, dtype=np.float64)
    for i in range(num):
        base = 2 * i * interval
        for j in range(interval):
            idx0 = base + j
            idx1 = idx0 + interval
            v0, v1 = _element_update_left(
                L_right[idx0],
                L_right[idx1],
                R_left[idx0],
                R_left[idx1],
                alpha,
            )
            out[idx0] = v0
            out[idx1] = v1
    return out


def _bp_update_right(L_left, R_right, layer, alpha):
    """整列右向更新，layer 为 1..n"""
    N = len(L_left)
    interval = 1 << (layer - 1)
    num = N // (2 * interval)
    out = np.zeros(N, dtype=np.float64)
    for i in range(num):
        base = 2 * i * interval
        for j in range(interval):
            idx0 = base + j
            idx1 = idx0 + interval
            v0, v1 = _element_update_right(
                L_left[idx0],
                L_left[idx1],
                R_right[idx0],
                R_right[idx1],
                alpha,
            )
            out[idx0] = v0
            out[idx1] = v1
    return out


class BPDecoder:
    """
    BP 译码器。
    L[i][j]：左向消息；R[i][j]：右向消息；列 0 为信源端，列 n 为信道端。
    """

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.info_idx = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, num_iters
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n
        alpha = self.alpha

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.info_idx, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for i in range(n):
                L[:, n - i - 1] = _bp_update_left(L[:, n - i], R[:, n - i - 1], n - i, alpha)

            for i in range(n):
                R[:, i + 1] = _bp_update_right(L[:, i + 1], R[:, i], i + 1, alpha)

            for idx in range(N):
                if self.frozen_bits[idx]:
                    u_hat[idx] = 0
                else:
                    u_hat[idx] = 0 if (L[idx, 0] + R[idx, 0]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break
            num_iters = it

        for idx in range(N):
            if self.frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if (L[idx, 0] + R[idx, 0]) >= 0 else 1

        return u_hat, num_iters
