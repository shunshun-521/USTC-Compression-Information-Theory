"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversed_index(x, n):
    """单索引比特倒序。"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError(f"N={N} must be a power of 2")
    return np.array([bit_reversed_index(i, n) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（Arikan 蝶形结构，O(N log N)）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字，满足 x = u @ G_N
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError(f"Length of u must be a power of 2, got {N}")

    block = N
    while block > 1:
        half = block // 2
        for p in range(0, N, block):
            for k in range(half):
                left = p + k
                u[left] = (u[left] + u[left + half]) % 2
        block = half
    return u


def build_generator_matrix(N):
    """构造 G_N = F^{\\otimes n}（Arikan 核 [[1,1],[0,1]] 的 Kronecker 积）"""
    n = int(np.log2(N))
    F = np.array([[1, 1], [0, 1]], dtype=int)

    def f_power(k):
        if k == 0:
            return np.array([[1]], dtype=int)
        return np.kron(f_power(k - 1), F)

    return f_power(n)


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("polar_encode([1,0,1,1]) =", x)
    G = build_generator_matrix(4)
    print("matrix encode =", (u @ G) % 2)
