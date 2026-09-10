"""
极化码 BP（置信传播）译码器
基于 Tanner 图（校验矩阵 H），使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode, polar_encode_matrix, bit_reversal_permutation
from channel import hard_decision_llr


def _build_parity_matrix(N, frozen_bits):
    """
    构造校验矩阵 H：对冻结位 f，约束 sum_j x[j]*G[f,j] = 0 (mod 2)
    其中 G = B_N F^{⊗n} 为极化码生成矩阵。
    """
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    frozen_idx = np.where(frozen_bits)[0]
    n = int(math.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    br = bit_reversal_permutation(N)
    B = np.eye(N, dtype=int)[br]
    GN = (B @ G) % 2
    H = GN[frozen_idx, :]
    return H.astype(np.int8)


class BPDecoder:
    """BP 译码器（Tanner 图 min-sum）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.H = _build_parity_matrix(N, frozen_bits)
        self.M, self.Nv = self.H.shape
        self._build_adjacency()

    def _build_adjacency(self):
        """预计算校验矩阵邻接关系"""
        self.var_to_chk = [[] for _ in range(self.Nv)]
        self.chk_to_var = [[] for _ in range(self.M)]
        for m in range(self.M):
            for v in range(self.Nv):
                if self.H[m, v]:
                    self.var_to_chk[v].append(m)
                    self.chk_to_var[m].append(v)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.Nv
        M = self.M

        Lq = np.tile(llr_ch, (M, 1))
        Lr = np.zeros((M, N), dtype=np.float64)

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for m in range(M):
                vars_m = self.chk_to_var[m]
                signs = 1.0
                min1, min2 = np.inf, np.inf
                min1_idx = vars_m[0]
                for v in vars_m:
                    q = Lq[m, v] - Lr[m, v]
                    signs *= np.sign(q) if q != 0 else 1.0
                    aq = abs(q)
                    if aq < min1:
                        min2 = min1
                        min1 = aq
                        min1_idx = v
                    elif aq < min2:
                        min2 = aq
                for v in vars_m:
                    q = Lq[m, v] - Lr[m, v]
                    mag = min2 if v == min1_idx else min1
                    out_sign = signs * (np.sign(q) if q != 0 else 1.0)
                    Lr[m, v] = self.alpha * out_sign * mag

            LQ = llr_ch + np.sum(Lr, axis=0)
            x_hat = (LQ < 0).astype(int)

            syndrome = np.mod(self.H @ x_hat, 2)
            if np.all(syndrome == 0):
                num_iters = it
                u_hat = self._x_to_u(x_hat)
                break

            for m in range(M):
                for v in self.chk_to_var[m]:
                    Lq[m, v] = LQ[v]

        LQ = llr_ch + np.sum(Lr, axis=0)
        x_hat = (LQ < 0).astype(int)
        u_hat = self._x_to_u(x_hat)
        return u_hat, num_iters

    def _x_to_u(self, x_hat):
        """从码字恢复源序列：u = x @ G^T (mod 2)"""
        n = self.n
        F = np.array([[1, 0], [1, 1]], dtype=int)
        G = F.copy()
        for _ in range(n - 1):
            G = np.kron(G, F)
        br = bit_reversal_permutation(self.N)
        B = np.eye(self.N, dtype=int)[br]
        GN = (B @ G) % 2
        u_hat = (x_hat @ GN) % 2
        u_hat[self.frozen_bits] = 0
        return u_hat.astype(int)
