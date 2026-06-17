"""
极化码编码器
编码：x = u * G_N，G_N = F^{⊗ n}，蝶形块 XOR 结构 O(N log N)
"""
import numpy as np


def bit_reversal_permutation(N):
  """返回长度 N 的比特倒序置换索引数组"""
  n = int(np.log2(N))
  if 2**n != N:
    raise ValueError("N must be a power of 2")
  return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
  """
  极化码编码（Arikan 蝶形，等价于 x = u * F^{⊗ n}）。

  参数：
      u: 长度为 N 的源序列（信息位 + 冻结位）

  返回：
      x: 长度为 N 的码字
  """
  u = np.asarray(u, dtype=int).copy()
  N = len(u)
  n = int(np.log2(N))
  if 2**n != N:
    raise ValueError("N must be a power of 2")

  for layer in range(1, n + 1):
    block = 2**layer
    half = block // 2
    for start in range(0, N, block):
      u[start : start + half] ^= u[start + half : start + block]
  return u


def build_generator_matrix(N):
  """构造 G_N = F^{⊗ n}（用于验证）"""
  F = np.array([[1, 0], [1, 1]], dtype=int)
  G = F.copy()
  while G.shape[0] < N:
    G = np.kron(G, F)
  return G
