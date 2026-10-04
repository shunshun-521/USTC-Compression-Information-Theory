"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=np.int64)
    rev_bits = (idx[:, None] >> np.arange(n - 1, -1, -1)) & 1
    powers = 2 ** np.arange(n - 1, -1, -1)
    return (rev_bits * powers).sum(axis=1)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    蝶形：F = [[1,1],[0,1]]，每层 (a,b)->(a xor b, b)
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be power of 2")
    n = int(np.log2(N))
    step = 1
    for _ in range(n):
        for i in range(0, N, 2 * step):
            for j in range(i, i + step):
                u[j] ^= u[j + step]
        step *= 2
    return u


def arikan_generator(N):
    """生成矩阵 F^{\\otimes n}（与 in-place 蝶形编码一致）"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = F.copy()
    while G.shape[0] < N:
        G = np.kron(G, F)
    return G


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = arikan_generator(4)
    x_mat = (u @ G) % 2
    print("butterfly+br:", x)
    print("matrix:", x_mat)
