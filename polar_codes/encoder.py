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
    极化码编码（蝶形结构，O(N log N)）。

    利用 Arikan 核 F=[[1,1],[0,1]]^{\\otimes n} 的蝶形运算；
    SC 译码器在比特倒序调度下与之匹配（等效于含 B_N 的生成矩阵）。
    """
    x = np.asarray(u, dtype=np.int8).copy()
    n = len(x)
    block = n
    while block > 1:
        half = block // 2
        for base in range(0, n, block):
            for k in range(half):
                idx = base + k
                x[idx] ^= x[idx + half]
        block = half
    return x


def polar_generator_matrix(N):
    """构造极化码生成矩阵 F^{\\otimes n}。"""
    f = np.array([[1, 1], [0, 1]], dtype=np.int8)
    g = f.copy()
    n = int(np.log2(N))
    for _ in range(n - 1):
        g = np.kron(g, f)
    return g


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    g = polar_generator_matrix(4)
    print("polar_encode:", x)
    print("matrix encode:", (u @ g) % 2)
