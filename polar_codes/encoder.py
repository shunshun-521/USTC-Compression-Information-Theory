"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def _butterfly_encode(v):
    """蝶形编码： (u[i], u[i+step]) -> (u[i] XOR u[i+step], u[i+step])"""
    v = np.asarray(v, dtype=np.int8).copy()
    N = len(v)
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            for j in range(step):
                v[i + j] ^= v[i + j + step]
        step <<= 1
    return v


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    等价于 x = u * G_N (mod 2)，G_N = B_N F^{\\otimes n}。
    实现为对 u 的比特倒序序列做蝶形编码。
    """
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    br = bit_reversal_permutation(N)
    return _butterfly_encode(u[br])


def polar_generator_matrix(N):
    """返回 N x N 生成矩阵 G_N = B_N F^{\\otimes n} (mod 2)。"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = np.array([[1]], dtype=np.int8)
    for _ in range(int(np.log2(N))):
        G = np.kron(G, F)
    br = bit_reversal_permutation(N)
    return G[br, :]


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix(4)
    x_mat = (u @ G) % 2
    print("polar_encode([1,0,1,1]) =", x)
    print("matrix encode =", x_mat)
    assert np.array_equal(x, x_mat)
