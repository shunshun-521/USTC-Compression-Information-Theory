"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.arange(N, dtype=int)
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(format(i, f"0{n}b")[::-1], 2)
    return rev


def polar_encode(u):
    """
    极化码编码（蝶形结构，O(N log N)）。

    实现 u -> x = u * F^⊗n（蝶形 XOR），与 SC 译码器内部比特倒序索引配套。
    等价于 x = (u * F^⊗n) * B_N，其中 B_N 倒序置换由译码端信道 LLR 索引承担。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n_block = N
    for _ in range(int(np.log2(N))):
        if n_block == 1:
            break
        n_split = n_block // 2
        for p in range(0, N, n_block):
            for k in range(n_split):
                l = p + k
                u[l] ^= u[l + n_split]
        n_block = n_split
    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u)
    print("x =", x)
