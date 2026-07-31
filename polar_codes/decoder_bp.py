"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器（极化码因子图，min-sum）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self._inv_brp = np.argsort(bit_reversal_permutation(N))
        self._large = 1e6

    def _f_min_sum(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, num_iters)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n

        # 因子图：L[i,s] 左向消息，R[i,s] 右向消息，s=0..n
        L_msg = np.zeros((N, n + 1), dtype=np.float64)
        R_msg = np.zeros((N, n + 1), dtype=np.float64)

        # 信道 LLR 映射到 SC 树顺序
        llr_internal = llr_ch[self._inv_brp]
        L_msg[:, n] = llr_internal
        R_msg[:, 0] = 0.0
        R_msg[self.frozen_bits, 0] = self._large

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            # 右到左更新 L
            for s in range(n, 0, -1):
                step = 2 ** (s - 1)
                block = 2 ** s
                for base in range(0, N, block):
                    for j in range(step):
                        i = base + j
                        ip = i + step
                        t = R_msg[i, s - 1] + L_msg[ip, s]
                        L_msg[i, s - 1] = self._f_min_sum(R_msg[i, s - 1] + L_msg[ip, s], L_msg[i, s])
                        L_msg[ip, s - 1] = self._f_min_sum(R_msg[i, s - 1], L_msg[i, s]) + L_msg[ip, s]

            # 左到右更新 R
            for s in range(1, n + 1):
                step = 2 ** (s - 1)
                block = 2 ** s
                for base in range(0, N, block):
                    for j in range(step):
                        i = base + j
                        ip = i + step
                        R_msg[i, s - 1] = self._f_min_sum(
                            R_msg[ip, s - 1] + L_msg[ip, s], R_msg[i, s - 1]
                        )
                        R_msg[ip, s - 1] = self._f_min_sum(R_msg[i, s - 1], L_msg[i, s]) + R_msg[ip, s - 1]

            # 早停检查
            total_llr = L_msg[:, 0] + R_msg[:, 0]
            u_internal = (total_llr < 0).astype(int)
            u_internal[self.frozen_bits] = 0
            x_hat = polar_encode(u_internal)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                u_hat = u_internal
                num_iters = it
                break

            u_hat = u_internal
            num_iters = it

        return u_hat, num_iters
