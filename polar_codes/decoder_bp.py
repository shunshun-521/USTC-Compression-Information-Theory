"""
极化码 BP（置信传播）译码器
基于极化码校验矩阵的 min-sum BP，含早停机制
"""
import numpy as np

from encoder import build_generator_matrix, polar_encode


def _gf2_inverse(G):
  N = G.shape[0]
  A = np.hstack([G.copy(), np.eye(N, dtype=np.int8)])
  for col in range(N):
    pivot = None
    for row in range(col, N):
      if A[row, col] == 1:
        pivot = row
        break
    if pivot is None:
      raise ValueError("Singular matrix")
    if pivot != col:
      A[[col, pivot]] = A[[pivot, col]]
    for row in range(N):
      if row != col and A[row, col] == 1:
        A[row] ^= A[col]
  return A[:, N:].astype(np.int8)


def _build_parity_matrix(N, frozen_bits):
  G = build_generator_matrix(N)
  G_inv = _gf2_inverse(G)
  frozen_idx = np.where(frozen_bits)[0]
  return G_inv[frozen_idx, :], G_inv


class BPDecoder:
  """BP 译码器（校验矩阵 min-sum）。"""

  def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
    self.N = N
    self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
    self.max_iter = max_iter
    self.alpha = alpha
    self.H, self.G_inv = _build_parity_matrix(N, self.frozen_bits)
    self.M = self.H.shape[0]
    self.cn_edges = [np.where(self.H[m])[0] for m in range(self.M)]
    self.vn_edges = [np.where(self.H[:, n])[0] for n in range(self.N)]

  def _codeword_to_source(self, x_hat):
    u = (x_hat @ self.G_inv) % 2
    u[self.frozen_bits] = 0
    return u.astype(np.int8)

  def decode(self, llr_ch):
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = self.N
    alpha = self.alpha

    Lq = np.tile(llr_ch, (self.M, 1))
    Rcv = np.zeros((self.M, N), dtype=np.float64)
    num_iters = self.max_iter

    for it in range(1, self.max_iter + 1):
      for m in range(self.M):
        cols = self.cn_edges[m]
        msgs = Lq[m, cols] - Rcv[m, cols]
        for idx, n in enumerate(cols):
          others = [msgs[j] for j in range(len(cols)) if j != idx]
          if not others:
            Rcv[m, n] = 0.0
          else:
            prod_sign = np.prod(np.sign(others))
            min_abs = np.min(np.abs(others))
            Rcv[m, n] = alpha * prod_sign * min_abs

      L_post = llr_ch.copy()
      for n in range(N):
        L_post[n] += np.sum(Rcv[:, n])

      for n in range(N):
        for m in self.vn_edges[n]:
          Lq[m, n] = L_post[n] - Rcv[m, n]

      x_hat = (L_post < 0).astype(np.int8)
      hard_ch = (llr_ch < 0).astype(np.int8)
      if np.array_equal(x_hat, hard_ch):
        num_iters = it
        break

    L_post = llr_ch.copy()
    for n in range(N):
      L_post[n] += np.sum(Rcv[:, n])
    x_hat = (L_post < 0).astype(np.int8)
    return self._codeword_to_source(x_hat), num_iters
