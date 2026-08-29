r"""
极化码编码器
编码：x = u * G_N，G_N = F^{\otimes n}，蝶形结构 O(N log N)
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(format(i, f"0{n}b")[::-1], 2)
    return rev


def polar_encode(u):
    """
    极化码编码（Arikan 蝶形结构，G_N = F^{\otimes n}）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = N
    if N & (N - 1):
        raise ValueError("Length of u must be a power of 2")

    while n > 1:
        n_split = n // 2
        for p in range(0, N, n):
            for k in range(n_split):
                idx = p + k
                u[idx] ^= u[idx + n_split]
        n = n_split
    return u


def polar_encode_with_br(u):
    """含比特倒序置换的编码（部分文献约定）"""
    x = polar_encode(u)
    return x[bit_reversal_permutation(len(u))]


def polar_generator_matrix(N):
    """生成 G_N = F^{\otimes n}"""
    F = np.array([[1, 1], [0, 1]], dtype=np.int8)
    G = F.copy()
    for _ in range(int(np.log2(N)) - 1):
        G = np.kron(G, F)
    return G


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    G = polar_generator_matrix(4)
    x = polar_encode(u)
    print("u =", u)
    print("x (butterfly) =", x)
    print("x (matrix u@G) =", (u @ G) % 2)
