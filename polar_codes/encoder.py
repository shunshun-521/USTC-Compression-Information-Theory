"""
极化码编码器
编码：x = G_N * u，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字

    实现：蝶形结构 x = G_N @ u (mod 2)，G_N = F^{\\otimes n}，F=[[1,1],[0,1]]
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))

    for layer in range(n):
        step = 1 << layer
        for i in range(0, N, 2 * step):
            for j in range(i, i + step):
                u[j] ^= u[j + step]

    return u


def polar_encode_matrix(u):
    """基于生成矩阵的参考编码（用于验证）"""
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    n = int(np.log2(N))
    F = np.array([[1, 1], [0, 1]], dtype=np.int8)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    return (G @ u) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print(f"u={u} -> x={x} (matrix={polar_encode_matrix(u)})")
    u2 = np.array([0, 1, 0, 1])
    x2 = polar_encode(u2)
    print(f"u={u2} -> x={x2} (matrix={polar_encode_matrix(u2)})")
