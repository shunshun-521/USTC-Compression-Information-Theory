"""
极化码编码器
编码：x = u * F_N（Kronecker 积），蝶形 XOR，O(N log N)
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组（部分构造/对照使用）。"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（与标准 Kronecker F^{⊗n} 一致，不做额外比特倒序）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    step = 1
    while step < N:
        for start in range(0, N, 2 * step):
            u[start: start + step] ^= u[start + step: start + 2 * step]
        step *= 2
    return u.astype(int)


def polar_generator_matrix(N):
    """生成矩阵 G_N = F^{⊗n}（GF(2)）。"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    while G.shape[0] < N:
        G = np.kron(G, F)
    return G


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix(4)
    print("encode:", x)
    print("matrix:", (u @ G) % 2)
