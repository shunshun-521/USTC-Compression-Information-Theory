"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode
from decoder_sc import f_operation, channel_llr_to_decoder


def _bp_update_left(left_col, right_col, layer):
    """从右向左更新 L 消息（列 layer）"""
    N = left_col.size
    interval = 2 ** (layer - 1)
    num = N // (interval * 2)
    out = np.zeros(N, dtype=np.float64)
    alpha = 0.9375
    for i in range(num):
        for j in range(interval):
            a = 2 * i * interval + j
            b = a + interval
            l0, l1 = left_col[a], left_col[b]
            r0, r1 = right_col[a], right_col[b]
            out[a] = alpha * f_operation(np.array([r1 + l1]), np.array([l0]))[0]
            out[b] = alpha * f_operation(np.array([l0]), np.array([r0]))[0] + l1
    return out


def _bp_update_right(left_col, right_col, layer):
    """从左向右更新 R 消息（列 layer-1 -> layer）"""
    N = left_col.size
    interval = 2 ** (layer - 1)
    num = N // (interval * 2)
    out = np.zeros(N, dtype=np.float64)
    alpha = 0.9375
    for i in range(num):
        for j in range(interval):
            a = 2 * i * interval + j
            b = a + interval
            l0, l1 = left_col[a], left_col[b]
            r0, r1 = right_col[a], right_col[b]
            out[a] = alpha * f_operation(np.array([r1 + l1]), np.array([r0]))[0]
            out[b] = alpha * f_operation(np.array([r0]), np.array([l0]))[0] + r1
    return out


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr = channel_llr_to_decoder(llr_ch)
        N, n = self.N, self.n
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr
        for i in range(N):
            if self.frozen_bits[i]:
                R[i, 0] = self.large
            else:
                R[i, 0] = 0.0

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)
        for it in range(1, self.max_iter + 1):
            num_iters = it
            for j in range(n, 0, -1):
                L[:, j - 1] = _bp_update_left(L[:, j], R[:, j - 1], j)
            for j in range(0, n):
                R[:, j + 1] = _bp_update_right(L[:, j + 1], R[:, j], j + 1)

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits == 1] = 0

            x_hat = polar_encode(u_hat)
            hard_x = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits == 1] = 0
        return u_hat, num_iters
