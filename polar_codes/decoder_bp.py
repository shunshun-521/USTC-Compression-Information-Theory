"""
极化码 BP（置信传播）译码器
基于校验矩阵 H 的 min-sum BP，含早停机制
"""
import numpy as np

from encoder import polar_encode, bit_reversal_permutation


def _gf2_inverse(G):
    n = G.shape[0]
    aug = np.concatenate([G.copy() % 2, np.eye(n, dtype=int)], axis=1)
    row = 0
    for col in range(n):
        pivot = next((r for r in range(row, n) if aug[r, col]), None)
        if pivot is None:
            continue
        if pivot != row:
            aug[[row, pivot]] = aug[[pivot, row]]
        for r in range(n):
            if r != row and aug[r, col]:
                aug[r] ^= aug[row]
        row += 1
    return aug[:, n:] % 2


def _gen_matrix(N):
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(int(np.log2(N)) - 1):
        G = np.kron(G, F)
    B = np.zeros((N, N), dtype=int)
    for i, j in enumerate(bit_reversal_permutation(N)):
        B[i, j] = 1
    return (B @ G) % 2


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """基于校验矩阵的 min-sum BP 译码器"""

    _cache = {}

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self._init_parity()

    def _init_parity(self):
        key = (self.N, tuple(self.frozen_bits.tolist()))
        if key not in BPDecoder._cache:
            G = _gen_matrix(self.N)
            Ginv = _gf2_inverse(G)
            frozen_idx = np.where(self.frozen_bits)[0]
            H = Ginv[frozen_idx, :]
            cn = [np.where(H[m])[0] for m in range(H.shape[0])]
            vn = [np.where(H[:, j])[0] for j in range(self.N)]
            BPDecoder._cache[key] = (H, frozen_idx, cn, vn)
        self.H, self.frozen_idx, self.cn_neighbors, self.vn_neighbors = BPDecoder._cache[key]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        M = self.H.shape[0]
        alpha = self.alpha

        var_llr = llr_ch.copy()
        check_to_var = np.zeros((M, N))
        num_iters = 0

        for it in range(1, self.max_iter + 1):
            var_to_check = np.zeros((M, N))
            for m, cols in enumerate(self.cn_neighbors):
                msgs = var_llr[cols]
                for j_idx, j in enumerate(cols):
                    others = np.delete(msgs, j_idx)
                    var_to_check[m, j] = others[0] if len(others) == 1 else np.sign(others).prod() * np.abs(others).min() * alpha if len(others) > 0 else msgs[0]

            for j, checks in enumerate(self.vn_neighbors):
                var_llr[j] = llr_ch[j] + check_to_var[checks, j].sum()

            for m, cols in enumerate(self.cn_neighbors):
                msgs = var_to_check[m, cols]
                for j_idx, j in enumerate(cols):
                    others = np.delete(msgs, j_idx)
                    check_to_var[m, j] = others[0] if len(others) == 1 else np.sign(others).prod() * np.abs(others).min() * alpha if len(others) > 0 else msgs[0]

            num_iters = it
            if np.all((self.H @ (var_llr < 0).astype(int)) % 2 == 0):
                break

        x_hat = (var_llr < 0).astype(int)
        u_hat = (x_hat @ _gf2_inverse(_gen_matrix(N))) % 2
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
