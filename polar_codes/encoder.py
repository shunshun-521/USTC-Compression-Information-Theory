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
  极化码编码（蝶形 XOR，与标准 F^{\\otimes n} 生成矩阵一致）。
  """
  u = np.asarray(u, dtype=np.int32).copy()
  N = len(u)
  n = int(np.log2(N))
  for layer in range(1, n + 1):
    block = 1 << layer
    half = block // 2
    for b in range(N // block):
      base = b * block
      u[base : base + half] ^= u[base + half : base + block]
  return u


if __name__ == "__main__":
  u = np.array([1, 0, 1, 1])
  x = polar_encode(u)
  print("u=", u, "-> x=", x)
  assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
  print("Encoder test passed.")
