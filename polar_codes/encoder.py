"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
  """返回长度 N 的比特倒序置换索引数组"""
  n = int(np.log2(N))
  return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
  """
  极化码编码（蝶形结构，与 Permuted SC 译码器配套）。

  参数：
      u: 长度为 N 的源序列（信息位 + 冻结位）

  返回：
      x: 长度为 N 的码字
  """
  u = np.asarray(u, dtype=int).copy()
  N = len(u)
  stage = N
  while stage > 1:
    split = stage // 2
    for p in range(0, N, stage):
      for k in range(split):
        idx = p + k
        u[idx] ^= u[idx + split]
    stage = split
  return u


def build_generator_matrix(N):
  """构造生成矩阵 G_N = F^{\\otimes n}，F=[[1,1],[0,1]]。"""
  F = np.array([[1, 1], [0, 1]], dtype=int)
  G = F.copy()
  n = int(np.log2(N))
  for _ in range(n - 1):
    G = np.kron(F, G) % 2
  return G
