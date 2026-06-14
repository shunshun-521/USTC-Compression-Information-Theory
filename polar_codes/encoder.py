"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def bit_reversed(i, n):
    """单索引比特倒序"""
    result = 0
    for bit in range(n):
        if i & (1 << bit):
            result |= 1 << (n - 1 - bit)
    return result


def _build_generator_matrix(N):
    """Arikan 生成矩阵 F^{\\otimes n}，F = [[1,1],[0,1]]"""
    F = np.array([[1, 1], [0, 1]], dtype=np.int8)
    G = np.array([[1]], dtype=np.int8)
    while G.shape[0] < N:
        G = np.kron(G, F)
    return G


def polar_encode(u):
    """
    极化码编码（蝶形 XOR，与 F^{\\otimes n} 等价）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    block = N
    for _ in range(n):
        half = block // 2
        for start in range(0, N, block):
            for i in range(half):
                idx = start + i
                u[idx] ^= u[idx + half]
        block = half
    return u.astype(int)


def polar_encode_matrix(u):
    """矩阵乘法编码，用于校验"""
    u = np.asarray(u, dtype=np.int8)
    G = _build_generator_matrix(len(u))
    return (u @ G) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    x_mat = polar_encode_matrix(u)
    print("butterfly:", x)
    print("matrix:", x_mat)
    assert np.array_equal(x, x_mat), "butterfly != matrix"
