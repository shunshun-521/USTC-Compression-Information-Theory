"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
  """返回长度 N 的比特倒序置换索引数组"""
  n = int(np.log2(N))
  if n == 0:
    return np.array([0], dtype=int)
  return np.array(
    [int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int
  )


def bit_reversed(x, n):
  """对标量索引 x 做 n 位比特倒序。"""
  result = 0
  for i in range(n):
    if x & (1 << i):
      result |= 1 << (n - 1 - i)
  return result


def polar_encode(u):
  """
  极化码编码（含比特倒序置换）。
  蝶形：从大块到小块逐级 XOR，最后做比特倒序。
  """
  u = np.asarray(u, dtype=np.int8).copy()
  N = len(u)
  n = int(np.log2(N))
  block = N
  for _ in range(n):
    block //= 2
    for p in range(0, N, 2 * block):
      for k in range(block):
        u[p + k] ^= u[p + k + block]
  br = bit_reversal_permutation(N)
  return u[br]


def build_generator_matrix(N):
  """构建与 polar_encode 一致的生成矩阵。"""
  G = np.zeros((N, N), dtype=int)
  for i in range(N):
    e = np.zeros(N, dtype=int)
    e[i] = 1
    G[i] = polar_encode(e)
  return G
