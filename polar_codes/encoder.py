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
        rev[i] = int(f"{i:0{n}b}"[::-1], 2)
    return rev


def bit_reversed(x, n):
    """单索引比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def polar_encode(u):
    """
    极化码编码（Arikan 核，蝶形 XOR：左分支 ^= 右分支）。
    与 G_N @ u 一致，其中 G_N = F^{\\otimes n}，F=[[1,1],[0,1]]。
    """
    u = np.array(u, dtype=int).copy()
    N = len(u)
    n = N
    while n > 1:
        n_split = n // 2
        for p in range(0, N, n):
            for k in range(n_split):
                l = p + k
                u[l] ^= u[l + n_split]
        n = n_split
    return u


if __name__ == "__main__":
    def arikan_gen(n):
        F = np.array([[1, 1], [0, 1]])
        F_n = F.copy()
        for _ in range(n - 1):
            F_n = np.kron(F, F_n)
        return F_n

    def mat_encode(G, u):
        x = np.zeros(len(u), dtype=int)
        for i in range(len(u)):
            for j in range(len(u)):
                x[i] ^= G[i, j] * u[j]
        return x

    u = np.array([1, 0, 1, 1])
    G = arikan_gen(2)
    x_ref = mat_encode(G, u)
    x = polar_encode(u)
    print(f"u={u}, x={x}, G@u={x_ref}")
    assert np.array_equal(x, x_ref), f"编码器错误: {x} vs {x_ref}"
    print("Encoder test passed.")
