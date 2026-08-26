"""
极化码编码器
编码：x = u * F^(⊗n)，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(format(i, f'0{n}b')[::-1], 2)
    return rev


def polar_encode(u):
    """
    极化码编码（蝶形 XOR 结构，x = u * F^(⊗n)）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = N
    while n > 1:
        n_split = n // 2
        for p in range(0, N, n):
            for k in range(n_split):
                l = p + k
                u[l] ^= u[l + n_split]
        n = n_split
    return u.astype(int)


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u)
    print("x =", x)
    F = np.array([[1, 0], [1, 1]])
    G = F.copy()
    for _ in range(1):
        G = np.kron(G, F)
    x_mat = np.mod(u @ G, 2)
    assert np.array_equal(x, x_mat), f"编码器错误: {x} vs {x_mat}"
    print("编码器校验通过")
