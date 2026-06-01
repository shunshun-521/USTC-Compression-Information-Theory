"""
极化码 BP（置信传播）译码器
基于奇偶校验矩阵 H 的 min-sum BP，含早停
"""
import numpy as np

from channel import hard_decision_llr
from encoder import build_generator_matrix, polar_encode


def _gf2_inverse(G):
  """GF(2) 上求逆"""
  N = G.shape[0]
  aug = np.concatenate([G.copy() % 2, np.eye(N, dtype=int)], axis=1)
  m, _ = aug.shape
  row = 0
  for col in range(N):
    pivot = None
    for r in range(row, m):
      if aug[r, col]:
        pivot = r
        break
    if pivot is None:
      continue
    aug[[row, pivot]] = aug[[pivot, row]]
    for r in range(m):
      if r != row and aug[r, col]:
        aug[r] ^= aug[row]
    row += 1
  return aug[:, N:] % 2


def build_polar_parity_matrix(frozen_bits):
  """由冻结位构造奇偶校验矩阵 H（H @ x = 0 mod 2）"""
  frozen_bits = np.asarray(frozen_bits).astype(bool)
  N = len(frozen_bits)
  G = build_generator_matrix(N)
  A = _gf2_inverse(G)
  H = A.T
  return H[frozen_bits, :]


class BPDecoder:
  """
  BP 译码器：在码字 x 上运行 LDPC 型 min-sum BP。
  """

  LARGE = 1e6

  def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
    self.N = N
    self.n = int(np.log2(N))
    self.frozen_bits = np.asarray(frozen_bits).astype(bool)
    self.max_iter = max_iter
    self.alpha = alpha
    self.H = build_polar_parity_matrix(self.frozen_bits)
    self.M = self.H.shape[0]
    self._A = _gf2_inverse(build_generator_matrix(N)).astype(np.int32)

    # 预计算邻接表
    self._cn_to_vn = [np.where(self.H[c])[0] for c in range(self.M)]
    self._vn_to_cn = [np.where(self.H[:, v])[0] for v in range(N)]

  def f_ms(self, messages):
    """min-sum：排除无穷大后的符号积与最小幅度"""
    if len(messages) == 0:
      return 0.0
    signs = np.sign(messages)
    signs[signs == 0] = 1
    sign_prod = np.prod(signs)
    amp = np.min(np.abs(messages))
    return self.alpha * sign_prod * amp

  def decode(self, llr_ch):
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = self.N

    # Lr[c][v], Lq[v][c] 存为字典
    Lr = {}
    Lv = llr_ch.copy()
    num_iters = self.max_iter

    for it in range(self.max_iter):
      # 校验节点更新
      for c in range(self.M):
        vns = self._cn_to_vn[c]
        msgs = np.array([Lv[v] - Lr.get((c, v), 0.0) for v in vns])
        for idx, v in enumerate(vns):
          other = np.delete(msgs, idx)
          Lr[(c, v)] = self.f_ms(other)

      # 变量节点更新
      for v in range(N):
        cns = self._vn_to_cn[v]
        extrinsic = sum(Lr.get((c, v), 0.0) for c in cns)
        Lv[v] = llr_ch[v] + extrinsic

      x_hat = (Lv < 0).astype(int)
      syndrome = (self.H @ x_hat) % 2
      if np.all(syndrome == 0):
        num_iters = it + 1
        u_hat = (x_hat @ self._A) % 2
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters

      # 早停：重编码码字一致
      u_hat = (x_hat @ self._A) % 2
      u_hat[self.frozen_bits] = 0
      x_re = polar_encode(u_hat)
      if np.array_equal(x_re, hard_decision_llr(llr_ch)):
        num_iters = it + 1
        return u_hat, num_iters

    x_hat = (Lv < 0).astype(int)
    u_hat = (x_hat @ self._A) % 2
    u_hat[self.frozen_bits] = 0
    return u_hat, num_iters
