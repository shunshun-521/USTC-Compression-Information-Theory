"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation

LARGE = 1e6


def _bp_channel_llr(llr_ch):
    """BP 因子图使用自然序信道 LLR"""
    return np.asarray(llr_ch, dtype=np.float64)


class BPDecoder:
    """
    BP 译码器。
    因子图有 n+1 列（列 0 到列 n），每列 N 个节点。
    列 0：信源比特端；列 n：信道接收端。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        """
        参数：
            N: 码长
            frozen_bits: 长度 N 的 bool 数组
            max_iter: 最大迭代次数
            alpha: min-sum 修正因子（典型值 0.9375）
        """
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)
        self.inv_br = np.argsort(self.br)
        self.frozen_br = self.frozen_bits[self.br]

    def _f_min_sum(self, a, b):
        """带 alpha 修正的 min-sum f 运算"""
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        """
        主译码函数。

        参数：
            llr_ch: 长度 N 的信道接收 LLR（自然序，对应因子图最右列）

        返回：
            u_hat: 长度 N 的估计源序列（自然序）
            num_iters: 实际迭代次数
        """
        N = self.N
        n = self.n
        llr_br = _bp_channel_llr(llr_ch)

        # L[i][j]: 从右到左的消息; R[i][j]: 从左到右的消息
        # 使用字典或二维列表，索引 i 为节点（0..N-1），j 为层（0..n）
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        # 初始化
        for i in range(N):
            L[i, n] = llr_br[i]
            R[i, 0] = 0.0
            if self.frozen_br[i]:
                R[i, 0] = LARGE

        num_iters = 0
        u_br = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it

            # 从右到左更新 L（列 n 到 1）
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for block in range(0, N, 2 * s):
                    for k in range(s):
                        i1 = block + k
                        i2 = block + k + s
                        L[i1, j - 1] = self._f_min_sum(
                            R[i1, j] + L[i2, j], L[i1, j]
                        )
                        L[i2, j - 1] = self._f_min_sum(
                            R[i1, j], L[i1, j]
                        ) + L[i2, j]

            # 从左到右更新 R（列 0 到 n-1）
            for j in range(0, n):
                s = 1 << j
                for block in range(0, N, 2 * s):
                    for k in range(s):
                        i1 = block + k
                        i2 = block + k + s
                        R[i1, j + 1] = self._f_min_sum(
                            R[i2, j] + L[i2, j + 1], R[i1, j]
                        )
                        R[i2, j + 1] = self._f_min_sum(
                            R[i1, j], L[i1, j + 1]
                        ) + R[i2, j]

            # 判决与早停
            for i in range(N):
                total = L[i, 0] + R[i, 0]
                if self.frozen_br[i]:
                    u_br[i] = 0
                else:
                    u_br[i] = 0 if total >= 0 else 1

            u_nat = u_br[self.inv_br]
            x_hat = polar_encode(u_nat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        # 最终判决（自然序）
        u_nat = np.zeros(N, dtype=int)
        for i in range(N):
            idx_br = self.br[i]
            total = L[idx_br, 0] + R[idx_br, 0]
            if self.frozen_bits[i]:
                u_nat[i] = 0
            else:
                u_nat[i] = 0 if total >= 0 else 1

        return u_nat, num_iters


if __name__ == "__main__":
    from construction import ga_construction
    from channel import bpsk_modulate, compute_llr

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    sigma = 0.01
    bp = BPDecoder(N, frozen_bits)
    errors = 0
    for _ in range(20):
        payload = np.random.randint(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        from encoder import polar_encode
        x = polar_encode(u)
        y = bpsk_modulate(x) + np.random.normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        u_hat, iters = bp.decode(llr)
        if not np.array_equal(u_hat[info_idx], payload):
            errors += 1
    print(f"BP low-noise errors: {errors}/20")
