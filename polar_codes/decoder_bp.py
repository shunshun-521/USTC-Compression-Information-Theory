"""
极化码 BP（置信传播）译码器
基于奇偶校验矩阵的 min-sum BP，含早停机制
"""
import math

import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _gen_matrix(N):
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(int(math.log2(N)) - 1):
        G = np.kron(G, F)
    br = bit_reversal_permutation(N)
    return G[br, :]


def _nullspace_parity(G_k):
    """构造奇偶校验矩阵 H，满足 H @ G_k^T = 0 (mod 2)。"""
    G_k = G_k.astype(int) % 2
    K, N = G_k.shape
    A = G_k.T.copy()
    rows = []
    n = A.shape[0]
    aug = np.concatenate([A, np.eye(n, dtype=int)], axis=1)
    r = 0
    for c in range(K):
        pivot = None
        for i in range(r, n):
            if aug[i, c]:
                pivot = i
                break
        if pivot is None:
            continue
        if pivot != r:
            aug[[r, pivot]] = aug[[pivot, r]]
        for i in range(n):
            if i != r and aug[i, c]:
                aug[i] ^= aug[r]
        r += 1
    for i in range(r, n):
        if np.any(aug[i, :K]):
            continue
        rows.append(aug[i, K:])
    if not rows:
        return np.zeros((0, N), dtype=int)
    return np.array(rows, dtype=int) % 2


def _minsum_f(a, b, alpha):
    sa = np.sign(a)
    sb = np.sign(b)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return alpha * sa * sb * np.minimum(np.abs(a), np.abs(b))


_PCM_CACHE = {}


def _get_pcm(N, frozen_bits):
    key = (N, tuple(np.asarray(frozen_bits, dtype=bool).tolist()))
    if key not in _PCM_CACHE:
        G = _gen_matrix(N)
        info_idx = np.where(~np.asarray(frozen_bits, dtype=bool))[0]
        G_k = G[info_idx, :]
        H = _nullspace_parity(G_k)
        _PCM_CACHE[key] = (H, G)
    return _PCM_CACHE[key]


class BPDecoder:
    """
    BP 译码器（基于极化码奇偶校验矩阵的 LDPC min-sum BP）。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.H, self.G = _get_pcm(N, frozen_bits)
        self.M, self.N_vars = self.H.shape
        self.cn_edges = [np.where(self.H[m])[0] for m in range(self.M)]
        self.vn_edges = [np.where(self.H[:, v])[0] for v in range(self.N)]

    def _f(self, a, b):
        return _minsum_f(a, b, self.alpha)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, num_iters
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        Lq = llr_ch.copy()
        Rmq = { (m, v): 0.0 for m in range(self.M) for v in self.cn_edges[m] }

        num_iters = self.max_iter
        x_hat = (llr_ch < 0).astype(int)

        for it in range(1, self.max_iter + 1):
            Lr = np.zeros(self.M, dtype=np.float64)
            for m in range(self.M):
                edges = self.cn_edges[m]
                msgs = np.array([Lq[v] - Rmq[(m, v)] for v in edges])
                for idx, v in enumerate(edges):
                    others = np.delete(msgs, idx)
                    prod_sign = np.prod(np.sign(others))
                    prod_sign = 1.0 if prod_sign == 0 else prod_sign
                    min_abs = np.min(np.abs(others)) if len(others) else 0.0
                    Rmq[(m, v)] = self.alpha * prod_sign * min_abs

            for v in range(N):
                Lq[v] = llr_ch[v] + sum(Rmq[(m, v)] for m in self.vn_edges[v])

            for v in range(N):
                x_hat[v] = 0 if Lq[v] >= 0 else 1

            u_hat = (x_hat @ self.G) % 2
            if np.array_equal(polar_encode(u_hat), x_hat):
                num_iters = it
                break

        u_hat = (x_hat @ self.G) % 2
        return u_hat.astype(int), num_iters


if __name__ == "__main__":
    from construction import ga_construction
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
    from decoder_sc import sc_decode

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    sigma = eb_n0_to_sigma(8.0, K / N)

    err_bp = err_sc = 0
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = np.random.randint(0, 2, K)
        llr = compute_llr(
            bpsk_modulate(polar_encode(u)) + np.random.normal(0, sigma, N), sigma
        )
        u_bp, _ = BPDecoder(N, frozen).decode(llr)
        u_sc = sc_decode(llr, frozen)
        err_bp += np.sum(u[info_idx] != u_bp[info_idx])
        err_sc += np.sum(u[info_idx] != u_sc[info_idx])
    print(f"BP info bit errors: {err_bp}, SC errors: {err_sc}")
