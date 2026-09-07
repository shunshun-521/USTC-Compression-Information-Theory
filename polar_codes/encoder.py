"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        r = 0
        for bit in range(n):
            r |= ((i >> bit) & 1) << (n - 1 - bit)
        rev[i] = r
    return rev


def polar_encode(u):
    """
    极化码编码：蝶形结构 + 比特倒序置换。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))

    for stage in range(n):
        step = 2 ** stage
        for i in range(0, N, 2 * step):
            for j in range(i, i + step):
                u[j] ^= u[j + step]

    rev = bit_reversal_permutation(N)
    return u[rev]


def build_generator_matrix(N):
    """构建生成矩阵 G_N = F^{⊗n} B_N"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(1, int(np.log2(N))):
        G = np.kron(G, F)
    rev = bit_reversal_permutation(N)
    return G[:, rev]


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("polar_encode:", x)
    G = build_generator_matrix(4)
    x_mat = (u @ G) % 2
    print("matrix encode:", x_mat)
