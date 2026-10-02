"""
极化码编码器
编码：对 u 做 Kronecker 蝶形变换（与 G_N = F^{⊗n} 左乘 u 等价，不含额外输出倒序）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=int)
    rev = ((idx[:, None] & (1 << np.arange(n))) != 0).astype(int)
    rev = rev[:, ::-1]
    powers = 1 << np.arange(n)
    return (rev * powers).sum(axis=1)


def polar_encode(u):
    """
    极化码编码：O(N log N) 蝶形结构。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = N
    while n > 1:
        half = n // 2
        for base in range(0, N, n):
            for k in range(half):
                i = base + k
                u[i] ^= u[i + half]
        n = half
    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    print("u=", u, "x=", polar_encode(u))
