"""
极化码 BP（置信传播）译码器
在 (N-K) 个奇偶校验约束的 Tanner 图上做 min-sum BP，早停于有效码字
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation


def _build_generator(N):
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = np.array([[1]], dtype=int)
    n = int(np.log2(N))
    for _ in range(n):
        G = np.kron(G, F) % 2
    return G


def _gf2_inverse(A):
    A = A.copy().astype(int)
    n = A.shape[0]
    aug = np.concatenate([A, np.eye(n, dtype=int)], axis=1)
    row = 0
    for col in range(n):
        pivot = None
        for r in range(row, n):
            if aug[r, col]:
                pivot = r
                break
        if pivot is None:
            raise ValueError("Matrix not invertible in GF(2)")
        if pivot != row:
            aug[[row, pivot]] = aug[[pivot, row]]
        for r in range(n):
            if r != row and aug[r, col]:
                aug[r] ^= aug[row]
        row += 1
    return aug[:, n:] % 2


def _parity_matrix_from_generator(B):
    """由生成矩阵 B (K x N) 的行空间求奇偶校验矩阵 P ((N-K) x N)，满足 P @ B.T = 0 (mod 2)"""
    B = B.astype(int)
    K, N = B.shape
    M = B.copy()
    pivot_cols = []
    row = 0
    for col in range(N):
        if row >= K:
            break
        if not np.any(M[row:, col]):
            continue
        piv = row + int(np.argmax(M[row:, col]))
        if piv != row:
            M[[row, piv]] = M[[piv, row]]
        pivot_cols.append(col)
        for r in range(K):
            if r != row and M[r, col]:
                M[r] ^= M[row]
        row += 1
    free_cols = [c for c in range(N) if c not in pivot_cols]
    checks = []
    for f in free_cols:
        p = np.zeros(N, dtype=int)
        p[f] = 1
        for j, pc in enumerate(pivot_cols):
            if M[j, f]:
                p[pc] = 1
        checks.append(p)
    return np.array(checks, dtype=int) if checks else np.zeros((0, N), dtype=int)


def _minsum_f(a, b, alpha):
    return alpha * f_operation(a, b)


class BPDecoder:
    """BP 译码器"""

    _cache = {}

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.info_idx = np.where(~self.frozen_bits)[0]
        self.K = len(self.info_idx)
        self._large = 1e8

        key = (N, tuple(self.frozen_idx.tolist()))
        if key not in BPDecoder._cache:
            G = _build_generator(N)
            H = _gf2_inverse(G)
            basis = []
            for idx in self.info_idx:
                u = np.zeros(N, dtype=int)
                u[idx] = 1
                basis.append(polar_encode(u))
            B = np.array(basis, dtype=int)  # K x N
            P = _parity_matrix_from_generator(B)
            BPDecoder._cache[key] = (H, P)
        self.H, self.P = BPDecoder._cache[key]
        self.check_to_var = [np.where(self.P[m])[0] for m in range(self.P.shape[0])]
        self.var_to_check = [[] for _ in range(N)]
        for m, cols in enumerate(self.check_to_var):
            for v in cols:
                self.var_to_check[v].append(m)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        alpha = self.alpha
        n_checks = self.P.shape[0]

        Lc_v = {}
        Lv_c = {}
        for v in range(N):
            for m in self.var_to_check[v]:
                Lc_v[(v, m)] = 0.0
                Lv_c[(v, m)] = 0.0

        L_ch = llr_ch.copy()
        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for v in range(N):
                checks = self.var_to_check[v]
                total = L_ch[v] + sum(Lc_v[(v, m)] for m in checks)
                for m in checks:
                    Lv_c[(v, m)] = total - Lc_v[(v, m)]

            for m in range(n_checks):
                vars_m = self.check_to_var[m]
                for v in vars_m:
                    others = [Lv_c[(vo, m)] for vo in vars_m if vo != v]
                    if not others:
                        Lc_v[(v, m)] = self._large if hasattr(self, "_large") else 1e8
                        continue
                    sign = 1.0
                    mag = abs(others[0])
                    for o in others[1:]:
                        sign *= np.sign(o) if o != 0 else 1.0
                        mag = min(mag, abs(o))
                    Lc_v[(v, m)] = alpha * sign * mag

            x_post = np.zeros(N, dtype=np.float64)
            for v in range(N):
                x_post[v] = L_ch[v] + sum(
                    Lc_v[(v, m)] for m in self.var_to_check[v]
                )

            x_hat_hard = (x_post < 0).astype(int)
            u_hat = (self.H @ x_hat_hard) % 2
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_x = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                num_iters = it
                break

        x_post = np.zeros(N, dtype=np.float64)
        for v in range(N):
            x_post[v] = L_ch[v] + sum(Lc_v[(v, m)] for m in self.var_to_check[v])
        x_hat_hard = (x_post < 0).astype(int)
        u_hat = (self.H @ x_hat_hard) % 2
        u_hat[self.frozen_idx] = 0

        return u_hat, num_iters
