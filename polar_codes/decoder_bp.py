"""
极化码 BP（置信传播）译码器
在码字域校验矩阵 H 上进行 min-sum BP，再映射回源比特 u
"""
import numpy as np

from encoder import polar_encode, polar_generator_matrix


def _gf2_inverse_matrix(G):
    N = G.shape[0]
    A = np.concatenate([G.T % 2, np.eye(N, dtype=int)], axis=1)
    row = 0
    piv = []
    for col in range(N):
        sel = None
        for r in range(row, N):
            if A[r, col]:
                sel = r
                break
        if sel is None:
            raise ValueError("G not invertible")
        if sel != row:
            A[[row, sel]] = A[[sel, row]]
        piv.append(col)
        for r in range(N):
            if r != row and A[r, col]:
                A[r] ^= A[row]
        row += 1
    inv = A[:, N:]
    return inv.T % 2


def _parity_check_matrix(N, info_indices):
    G = polar_generator_matrix(N, apply_bit_reversal=False)
    G_sub = G[info_indices, :]
    A = G_sub.copy() % 2
    K, Nloc = A.shape
    pivot_cols = []
    row = 0
    for col in range(Nloc):
        if row >= K:
            break
        sel = next((r for r in range(row, K) if A[r, col]), None)
        if sel is None:
            continue
        if sel != row:
            A[[row, sel]] = A[[sel, row]]
        pivot_cols.append(col)
        for r in range(K):
            if r != row and A[r, col]:
                A[r] ^= A[row]
        row += 1
    free_cols = [c for c in range(Nloc) if c not in pivot_cols]
    H_rows = []
    for fc in free_cols:
        v = np.zeros(Nloc, dtype=int)
        v[fc] = 1
        for i, pc in enumerate(pivot_cols):
            if A[i, fc]:
                v[pc] = 1
        H_rows.append(v)
    return np.array(H_rows, dtype=int) if H_rows else np.zeros((0, Nloc), dtype=int)


class BPDecoder:
    """基于校验矩阵 H 的 min-sum BP（码字域）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.frozen = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.info_idx = np.where(~self.frozen)[0]
        G = polar_generator_matrix(N, apply_bit_reversal=False)
        self.Ginv = _gf2_inverse_matrix(G)
        self.H = _parity_check_matrix(N, self.info_idx)
        self.M = self.H.shape[0]
        self.var_to_chk = [np.where(self.H[:, j])[0] for j in range(N)]
        self.chk_to_var = [np.where(self.H[m, :])[0] for m in range(self.M)]

    def _x_to_u(self, x_hat):
        u_hat = (x_hat @ self.Ginv) % 2
        u_hat[self.frozen] = 0
        return u_hat.astype(int)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, M = self.N, self.M
        Lr = np.zeros((M, N), dtype=np.float64)
        channel = llr_ch.copy()

        def total_llr():
            t = channel.copy()
            for j in range(N):
                t[j] += np.sum(Lr[self.var_to_chk[j], j])
            return t

        def hard_x():
            return (total_llr() < 0).astype(int)

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for m in range(M):
                idx = self.chk_to_var[m]
                if len(idx) == 0:
                    continue
                msgs = []
                for j in idx:
                    msgs.append(channel[j] + np.sum(Lr[self.var_to_chk[j], j]) - Lr[m, j])
                for jpos, j in enumerate(idx):
                    other = [msgs[k] for k in range(len(idx)) if k != jpos]
                    if not other:
                        Lr[m, j] = 0.0
                        continue
                    signs = np.prod(np.sign(other))
                    mags = np.min(np.abs(other))
                    Lr[m, j] = self.alpha * signs * mags

            x_hat = hard_x()
            u_hat = self._x_to_u(x_hat)
            x_re = polar_encode(u_hat)
            x_hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_re, x_hard):
                num_iters = it
                break

        return self._x_to_u(hard_x()), num_iters
