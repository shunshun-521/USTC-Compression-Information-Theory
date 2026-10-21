"""
极化码 BP（置信传播）译码器
在生成矩阵定义的 Tanner 图上做 min-sum BP，含早停
"""
import math
import numpy as np
from decoder_sc import f_operation
from encoder import polar_encode
from channel import hard_decision_llr


def _build_generator_matrix(N):
    """G_N = F^{\\otimes n}（与默认 polar_encode 一致，无输出比特倒序）"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = F.copy()
    for _ in range(int(math.log2(N)) - 1):
        G = np.kron(G, F) % 2
    return G % 2


def _parity_check_matrix(G_sub):
    """由 K×N 生成矩阵 G_sub 构造 (N-K)×N 校验矩阵 H（G_sub @ h = 0）"""
    G = G_sub.astype(np.uint8)
    K, N = G.shape
    pivots = []
    row = 0
    for col in range(N):
        sel = None
        for r in range(row, K):
            if G[r, col]:
                sel = r
                break
        if sel is None:
            continue
        if sel != row:
            G[[row, sel]] = G[[sel, row]]
        for r in range(K):
            if r != row and G[r, col]:
                G[r] ^= G[row]
        pivots.append(col)
        row += 1
        if row == K:
            break
    free_cols = [c for c in range(N) if c not in pivots]
    H = np.zeros((len(free_cols), N), dtype=np.uint8)
    for i, fc in enumerate(free_cols):
        v = np.zeros(N, dtype=np.uint8)
        v[fc] = 1
        for j, pc in enumerate(pivots):
            if G[j, fc]:
                v[pc] = 1
        H[i] = v
    return H


def _gf2_inverse(A):
    n = A.shape[0]
    aug = np.concatenate([A.astype(np.uint8), np.eye(n, dtype=np.uint8)], axis=1)
    row = 0
    for col in range(n):
        sel = next((r for r in range(row, n) if aug[r, col]), None)
        if sel is None:
            raise ValueError("singular matrix")
        if sel != row:
            aug[[row, sel]] = aug[[sel, row]]
        for r in range(n):
            if r != row and aug[r, col]:
                aug[r] ^= aug[row]
        row += 1
    return aug[:, n:]


class BPDecoder:
    """基于校验矩阵 H 的 min-sum BP（变量为码字比特 x）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375, info_indices=None):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.info_indices = (
            np.asarray(info_indices, dtype=int)
            if info_indices is not None
            else np.where(~self.frozen_bits)[0]
        )
        G_full = _build_generator_matrix(N)
        self.G_inv = _gf2_inverse(G_full)
        G_sub = G_full[self.info_indices, :]
        self.H = _parity_check_matrix(G_sub)
        self.M, self.Nc = self.H.shape
        self._build_adjacency()

    def _build_adjacency(self):
        H = self.H
        self.var_to_chk = [np.where(H[:, v])[0] for v in range(self.Nc)]
        self.chk_to_var = [np.where(H[m])[0] for m in range(self.M)]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, M = self.Nc, self.M
        Lq = llr_ch.copy()
        Lr = np.zeros((M, N), dtype=np.float64)

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it
            # 校验节点更新
            for m in range(M):
                vars_m = self.chk_to_var[m]
                msgs = []
                for v in vars_m:
                    msgs.append(Lq[v] - Lr[m, v])
                for idx, v in enumerate(vars_m):
                    other = np.delete(msgs, idx)
                    if len(other) == 0:
                        Lr[m, v] = 0.0
                    else:
                        signs = np.prod(np.sign(other))
                        mags = np.min(np.abs(other))
                        Lr[m, v] = self.alpha * signs * mags

            # 变量节点更新
            for v in range(N):
                Lq[v] = llr_ch[v] + np.sum(Lr[:, v])

            x_hat = (Lq < 0).astype(int)
            u_hat = (x_hat @ self.G_inv) % 2
            u_hat[self.frozen_bits] = 0
            if np.array_equal(polar_encode(u_hat), x_hat):
                break

        x_hat = (Lq < 0).astype(int)
        u_hat = (x_hat @ self.G_inv) % 2
        u_hat[self.frozen_bits] = 0

        return u_hat, num_iters
