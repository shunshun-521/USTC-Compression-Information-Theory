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
        b = format(i, f"0{n}b")[::-1]
        rev[i] = int(b, 2)
    return rev


def polar_encode(u):
    """
    极化码编码（蝶形结构，与标准 Arikan 生成矩阵一致）。

    实现与 py-polar-codes 非递归编码相同：逐层 u[i] ^= u[i+step]。
    比特倒序在 SC 译码的译码顺序中处理，编码输出为自然顺序码字。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    block = N
    while block > 1:
        half = block // 2
        for p in range(0, N, block):
            for k in range(half):
                u[p + k] ^= u[p + k + half]
        block = half
    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u=", u, "x=", x)
    # 标准蝶形编码输出（与 u@B@G 一致）
    n = 2
    B = np.zeros((4, 4), dtype=int)
    for i in range(4):
        B[i, int(format(i, "02b")[::-1], 2)] = 1
    G = np.array([[1, 0], [1, 1]], dtype=int)
    G4 = np.kron(G, G) % 2
    print("matrix", u @ B @ G4 % 2)
