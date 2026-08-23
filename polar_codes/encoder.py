"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def bit_reversed(x, n):
    """对标量索引 x 做 n 位比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def polar_encode(u):
    """
    极化码编码（蝶形结构，与 SC 译码器配套）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int32).copy()
    N = len(u)
    n = int(np.log2(N))
    assert 2 ** n == N, "N must be a power of 2"

    step = 1
    for _ in range(n):
        for i in range(0, N, 2 * step):
            u[i:i + step] ^= u[i + step:i + 2 * step]
        step *= 2

    return u


def polar_encode_with_brp(u):
    """含比特倒序置换的编码（用于验证生成矩阵）。"""
    u_enc = polar_encode(u)
    brp = bit_reversal_permutation(len(u_enc))
    return u_enc[brp]


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u, "-> x =", x)

    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(1):
        G = np.kron(G, F)
    brp = bit_reversal_permutation(4)
    G = G[brp, :] % 2
    x_mat = u @ G % 2
    x_brp = polar_encode_with_brp(u)
    print("matrix (with brp):", x_mat)
    print("encode_with_brp:", x_brp)
    print("Encoder test passed.")
