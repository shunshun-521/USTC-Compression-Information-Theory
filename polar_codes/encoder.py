"""
极化码编码器
编码：x = bitrev(u * F_N)，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
  """返回长度 N 的比特倒序置换索引数组"""
  n = int(np.log2(N))
  return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def _butterfly_encode(u):
  """蝶形编码（不含比特倒序）。"""
  u = np.asarray(u, dtype=np.int8).copy()
  N = len(u)
  n = int(np.log2(N))
  step = N
  for _ in range(n):
    step //= 2
    for p in range(0, N, 2 * step):
      for k in range(step):
        u[p + k] ^= u[p + k + step]
  return u


def polar_encode(u):
  """
  极化码编码（含比特倒序置换）。
  """
  u = _butterfly_encode(u)
  return u[bit_reversal_permutation(len(u))]


def build_generator_matrix(N):
  """构建与 polar_encode 一致的生成矩阵，满足 x = u @ G (mod 2)。"""
  G = np.zeros((N, N), dtype=np.int8)
  for i in range(N):
    u = np.zeros(N, dtype=np.int8)
    u[i] = 1
    G[i] = polar_encode(u)
  return G


if __name__ == "__main__":
  u = np.array([1, 0, 1, 1])
  x = polar_encode(u)
  print("u =", u)
  print("x =", x)
  G = build_generator_matrix(4)
  print("G @ u.T =", (u @ G) % 2)
