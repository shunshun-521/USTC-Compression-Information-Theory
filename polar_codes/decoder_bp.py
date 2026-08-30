"""
极化码 BP（置信传播）译码器
基于校验矩阵 H（冻结位约束 x @ G 在冻结位置为 0），min-sum 近似，含早停
"""
import numpy as np
from encoder import polar_encode, polar_encode_matrix


class BPDecoder:
    """BP 译码器（min-sum LDPC on polar frozen-bit checks）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.G = polar_encode_matrix(N)
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]
        self._build_graph()

    def _build_graph(self):
        """冻结位 i 对应约束 (x @ G)_i = u_i = 0"""
        N = self.N
        self.check_vars = []
        for i in self.frozen_idx:
            self.check_vars.append(np.where(self.G[i, :] == 1)[0].tolist())
        self.var_checks = [[] for _ in range(N)]
        for ci, vlist in enumerate(self.check_vars):
            for v in vlist:
                self.var_checks[v].append(ci)

    @staticmethod
    def _cn_update(msgs, alpha):
        m = len(msgs)
        out = np.zeros(m)
        for i in range(m):
            prod_sign = 1.0
            min_abs = np.inf
            for j in range(m):
                if j == i:
                    continue
                prod_sign *= np.sign(msgs[j]) if msgs[j] != 0 else 1.0
                min_abs = min(min_abs, abs(msgs[j]))
            out[i] = alpha * prod_sign * (min_abs if np.isfinite(min_abs) else 0.0)
        return out

    def decode(self, llr_ch):
        """主译码函数。返回：u_hat, num_iters"""
        N = self.N
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n_checks = len(self.check_vars)

        q = {ci: {v: 0.0 for v in self.check_vars[ci]} for ci in range(n_checks)}
        r = {ci: {v: 0.0 for v in self.check_vars[ci]} for ci in range(n_checks)}

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            # 变量节点 -> 校验节点
            for ci in range(n_checks):
                for v in self.check_vars[ci]:
                    total = llr_ch[v]
                    for cj in self.var_checks[v]:
                        if cj != ci:
                            total -= r[cj][v]
                    q[ci][v] = total

            # 校验节点 -> 变量节点
            for ci in range(n_checks):
                vlist = self.check_vars[ci]
                msgs = np.array([q[ci][v] for v in vlist])
                out = self._cn_update(msgs, self.alpha)
                for v, val in zip(vlist, out):
                    r[ci][v] = val

            # 后验 LLR on codeword bits
            L_post = llr_ch.copy()
            for v in range(N):
                for ci in self.var_checks[v]:
                    L_post[v] += r[ci][v]

            x_soft = (L_post < 0).astype(int)
            u_hat = (x_soft @ self.G) % 2
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard):
                num_iters = it
                break

        L_post = llr_ch.copy()
        for v in range(N):
            for ci in self.var_checks[v]:
                L_post[v] += r[ci][v]
        x_soft = (L_post < 0).astype(int)
        u_hat = (x_soft @ self.G) % 2
        u_hat[self.frozen_idx] = 0

        return u_hat, num_iters
