"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np


def _gen_matrix(n):
    F = np.array([[1, 0], [1, 1]])
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    return G % 2


def _f_ms_vec(a, b, alpha):
    sa = np.sign(a)
    sb = np.sign(b)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return alpha * sa * sb * np.minimum(np.abs(a), np.abs(b))


def _bp_update_left(left_array, right_array, layer_n, alpha):
    N = len(left_array)
    interval = 1 << (layer_n - 1)
    num = N // (2 * interval)
    value = np.empty(N)
    for i in range(num):
        base = 2 * i * interval
        idx = np.arange(base, base + interval)
        l0 = left_array[idx]
        l1 = left_array[idx + interval]
        r0 = right_array[idx]
        r1 = right_array[idx + interval]
        value[idx] = _f_ms_vec(r1 + l1, l0, alpha)
        value[idx + interval] = _f_ms_vec(l0, r0, alpha) + l1
    return value


def _bp_update_right(left_array, right_array, layer_n, alpha):
    N = len(left_array)
    interval = 1 << (layer_n - 1)
    num = N // (2 * interval)
    value = np.empty(N)
    for i in range(num):
        base = 2 * i * interval
        idx = np.arange(base, base + interval)
        l0 = left_array[idx]
        l1 = left_array[idx + interval]
        r0 = right_array[idx]
        r1 = right_array[idx + interval]
        value[idx] = _f_ms_vec(r1 + l1, r0, alpha)
        value[idx + interval] = _f_ms_vec(l0, r0, alpha) + r1
    return value


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.G = _gen_matrix(self.n)
        self._inf = 1e9

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        left = np.zeros((N, n + 1))
        right = np.zeros((N, n + 1))
        left[:, n] = llr_ch
        right[:, 0] = np.where(self.frozen_bits, self._inf, 0.0)

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for i in range(n):
                left[:, n - i - 1] = _bp_update_left(
                    left[:, n - i], right[:, n - i - 1], n - i, self.alpha
                )
            for i in range(n):
                right[:, i + 1] = _bp_update_right(
                    left[:, i + 1], right[:, i], i + 1, self.alpha
                )

            u_llr = left[:, 0] + right[:, 0]
            u_hat = (u_llr < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_llr = left[:, n] + right[:, n]
            x_hat_hard = (x_llr < 0).astype(int)
            if np.array_equal((u_hat @ self.G) % 2, x_hat_hard):
                num_iters = it
                break

        u_llr = left[:, 0] + right[:, 0]
        u_hat = (u_llr < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
