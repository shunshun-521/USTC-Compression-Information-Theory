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


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    蝶形：(u[i], u[i+step]) -> (u[i] XOR u[i+step], u[i+step])
    即 u[i] ^= u[i+step]，共 log2(N) 层，最后比特倒序置换。
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

    br = bit_reversal_permutation(N)
    return u[br].astype(int)


def polar_generator_matrix(N):
    """构造 G_N = B_N F^{\\otimes n}（用于验证）"""
    n = int(np.log2(N))
    G = np.eye(1, dtype=int)
    F = np.array([[1, 1], [0, 1]], dtype=int)
    for _ in range(n):
        G = np.kron(G, F)
    br = bit_reversal_permutation(N)
    return G[br, :]


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix(4)
    x_ref = (u @ G) % 2
    print("u:", u)
    print("x (butterfly):", x)
    print("x (matrix):", x_ref)
