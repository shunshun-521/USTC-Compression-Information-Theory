"""
极化码编码器
编码：蝶形 XOR（右半加到左半，mod 2），与 mcba1n polar_encode 一致
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(f"{i:0{n}b}"[::-1], 2)
    return rev


def bit_reversed_index(i, n):
    """单索引比特倒序"""
    result = 0
    for b in range(n):
        if i & (1 << b):
            result |= 1 << (n - 1 - b)
    return result


def polar_encode(u):
    """
    极化码编码（非系统化，蝶形 XOR）。
    u[l] ^= u[l + n_split] 逐层减半，无输出比特倒序。
    """
    u = np.array(u, dtype=int).copy()
    n = len(u)
    while n > 1:
        n_split = n // 2
        for p in range(0, len(u), n):
            for k in range(n_split):
                u[p + k] ^= u[p + k + n_split]
        n = n_split
    return u


def polar_encode_generator_matrix(u):
    """生成矩阵编码（用于交叉验证）"""
    u = np.array(u, dtype=int)
    N = len(u)
    n = int(np.log2(N))
    F = np.array([[1, 1], [0, 1]], dtype=int)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    return (u @ G) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    print("polar_encode:", polar_encode(u))
    print("matrix:", polar_encode_generator_matrix(u))
