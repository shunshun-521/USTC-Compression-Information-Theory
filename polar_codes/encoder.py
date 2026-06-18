"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
  """返回长度 N 的比特倒序置换索引数组"""
  n = int(np.log2(N))
  return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
  """
  极化码编码（蝶形 XOR，与 Permuted SC 译码器配套）。
  """
  u = np.asarray(u, dtype=int).copy()
  n = len(u)
  block = n
  for _ in range(n):
    if block == 1:
      break
    half = block // 2
    for start in range(0, n, block):
      for k in range(half):
        idx = start + k
        u[idx] ^= u[idx + half]
    block = half
  return u


if __name__ == "__main__":
  u = np.array([1, 0, 1, 1])
  x = polar_encode(u)
  print("u =", u)
  print("x =", x)
  assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
  print("编码器校验通过")
