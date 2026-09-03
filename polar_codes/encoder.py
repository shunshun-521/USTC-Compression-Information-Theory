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
    x = np.arange(N, dtype=np.int64)
    br = np.zeros(N, dtype=np.int64)
    for bit in range(n):
        br |= ((x >> bit) & 1) << (n - 1 - bit)
    return br.astype(int)


def polar_encode(u):
    """
    极化码编码（Arikan 蝶形，无输出比特倒序）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    block = N
    for _ in range(n):
        half = block // 2
        for start in range(0, N, block):
            for k in range(half):
                idx = start + k
                u[idx] ^= u[idx + half]
        block = half
    return u


def build_generator_matrix(N):
    """构建 F^{\\otimes n} 生成矩阵，用于验证"""
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    return G


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("polar_encode([1,0,1,1]) =", x)
    G = build_generator_matrix(4)
    x_mat = (u @ G) % 2
    print("matrix encode =", x_mat)
