"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode, build_generator_matrix, bit_reversal_permutation


def _bp_f(x, y, alpha):
    """min-sum f 运算"""
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """
    BP 译码器。
    在极化码生成矩阵对应的因子图上执行 min-sum BP。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.info_idx = np.where(~self.frozen_bits)[0]
        self.G = build_generator_matrix(N)
        self.LARGE = 1e6
        self._build_graph()

    def _build_graph(self):
        """构建 VN-CN 邻接表：CN j 连接所有满足 G[i,j]=1 的 VN i"""
        N = self.N
        self.vn_to_cn = [[] for _ in range(N)]
        self.cn_to_vn = [[] for _ in range(N)]
        for j in range(N):
            for i in range(N):
                if self.G[i, j]:
                    self.vn_to_cn[i].append(j)
                    self.cn_to_vn[j].append(i)

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, num_iters"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N

        ch_llr = np.zeros(N, dtype=np.float64)
        ch_llr[:] = llr_ch

        vn_llr = np.zeros(N, dtype=np.float64)
        vn_llr[self.frozen_idx] = self.LARGE

        msg_vn_to_cn = np.zeros((N, N), dtype=np.float64)
        msg_cn_to_vn = np.zeros((N, N), dtype=np.float64)

        num_iters = 0
        for it in range(1, self.max_iter + 1):
            for j in range(N):
                connected = self.cn_to_vn[j]
                for i in connected:
                    others = [k for k in connected if k != i]
                    if not others:
                        msg_cn_to_vn[j, i] = ch_llr[j]
                    else:
                        prod = ch_llr[j]
                        for k in others:
                            prod = _bp_f(prod, msg_vn_to_cn[k, j], self.alpha)
                        msg_cn_to_vn[j, i] = prod

            for i in range(N):
                if i in self.frozen_idx:
                    continue
                for j in self.vn_to_cn[i]:
                    others = [k for k in self.vn_to_cn[i] if k != j]
                    total = vn_llr[i]
                    for k in others:
                        total += msg_cn_to_vn[k, i]
                    msg_vn_to_cn[i, j] = total

            total_llr = np.zeros(N, dtype=np.float64)
            for i in range(N):
                total_llr[i] = vn_llr[i] + np.sum(msg_cn_to_vn[self.vn_to_cn[i], i])

            u_hat = np.zeros(N, dtype=int)
            u_hat[self.info_idx] = (total_llr[self.info_idx] < 0).astype(int)

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                return u_hat, it
            num_iters = it

        total_llr = np.zeros(N, dtype=np.float64)
        for i in range(N):
            total_llr[i] = vn_llr[i] + np.sum(msg_cn_to_vn[self.vn_to_cn[i], i])
        u_hat = np.zeros(N, dtype=int)
        u_hat[self.info_idx] = (total_llr[self.info_idx] < 0).astype(int)
        return u_hat, num_iters
