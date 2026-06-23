"""
极化码 BP（置信传播）译码器
基于校验矩阵 H 的因子图，min-sum 近似，含早停机制
"""
import numpy as np

from decoder_sc import f_operation
from encoder import polar_encode, polar_generator_matrix
from channel import hard_decision_llr


def _gf2_inverse(G):
    """GF(2) 矩阵求逆"""
    G = G.copy() % 2
    N = len(G)
    aug = np.concatenate([G, np.eye(N, dtype=int)], axis=1) % 2
    row = 0
    for col in range(N):
        piv = next((r for r in range(row, N) if aug[r, col] == 1), None)
        if piv is None:
            raise ValueError("Matrix is singular over GF(2)")
        if piv != row:
            aug[[row, piv]] = aug[[piv, row]]
        for r in range(N):
            if r != row and aug[r, col] == 1:
                aug[r] = (aug[r] + aug[row]) % 2
        row += 1
    return aug[:, N:]


def _build_parity_matrix(N, frozen_bits):
    """构造校验矩阵 H：冻结位对应 G^{-1} 的行"""
    G = polar_generator_matrix(N)
    Ginv = _gf2_inverse(G)
    frozen_idx = np.where(np.asarray(frozen_bits, dtype=int) == 1)[0]
    return Ginv[frozen_idx, :] % 2


class BPDecoder:
    """基于校验矩阵 H 的 BP 译码器"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]
        self.H = _build_parity_matrix(N, frozen_bits)
        self.M, self.N = self.H.shape
        self.cn_edges = [np.where(self.H[m])[0] for m in range(self.M)]
        self.vn_edges = [np.where(self.H[:, n])[0] for n in range(self.N)]

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, num_iters"""
        N, M = self.N, self.M
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        # 从信道 LLR 恢复码字硬判决用于初值
        L_vn = llr_ch.copy()
        L_cn = np.zeros((M, N), dtype=np.float64)

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it

            # 校验节点更新（CN -> VN）
            for m in range(M):
                edges = self.cn_edges[m]
                if len(edges) < 2:
                    continue
                incoming = np.array(
                    [L_vn[v] - L_cn[m, v] for v in edges], dtype=np.float64
                )
                for k, v in enumerate(edges):
                    prod = np.delete(incoming, k)
                    msg = prod[0]
                    for val in prod[1:]:
                        msg = self._f_min_sum(msg, val)
                    L_cn[m, v] = msg

            # 变量节点更新（VN -> CN + 先验）
            for v in range(N):
                prior = llr_ch[v]
                cn_msgs = np.array([L_cn[m, v] for m in self.vn_edges[v]])
                total = prior + np.sum(cn_msgs)
                L_vn[v] = total
                for m in self.vn_edges[v]:
                    L_cn[m, v] = total - L_cn[m, v]

            # 冻结位先验
            for v in self.frozen_idx:
                L_vn[v] = self.LARGE

            # 从码字 LLR 恢复 u：x_hat = hard(llr_ch), u_hat = x_hat @ Ginv
            x_hat = hard_decision_llr(L_vn)
            G = polar_generator_matrix(N)
            Ginv = _gf2_inverse(G)
            u_hat = (x_hat @ Ginv) % 2
            u_hat[self.frozen_idx] = 0

            x_reenc = polar_encode(u_hat)
            if np.array_equal(x_reenc, hard_decision_llr(llr_ch)):
                break

        Ginv = _gf2_inverse(polar_generator_matrix(N))
        x_hat = hard_decision_llr(L_vn)
        u_hat = (x_hat @ Ginv) % 2
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters


if __name__ == "__main__":
    from construction import ga_construction
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(2)
    sigma = eb_n0_to_sigma(6.0, K / N)
    ok = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x) + rng.normal(0, sigma, N), sigma)
        u_hat, iters = BPDecoder(N, frozen_bits).decode(llr)
        if np.array_equal(u_hat, u):
            ok += 1
    print(f"BP test: {ok}/50 correct at Eb/N0=6dB")
