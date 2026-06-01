"""
极化码编码器
编码：u * G_N，蝶形结构 O(N log N)，与 SC 译码器配套（无输出比特倒序）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        r = 0
        v = i
        for _ in range(n):
            r = (r << 1) | (v & 1)
            v >>= 1
        rev[i] = r
    return rev


def bit_reversed_index(i, n):
    """单索引比特倒序"""
    r = 0
    for k in range(n):
        if i & (1 << k):
            r |= 1 << (n - 1 - k)
    return r


def polar_encode(u):
    """
    极化码编码（蝶形 XOR，与因子图自然序一致）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    x = np.asarray(u, dtype=int).copy()
    N = len(x)
    if N & (N - 1):
        raise ValueError("Length must be a power of 2")

    n = N
    for _ in range(int(np.log2(N))):
        n_split = n // 2
        if n_split == 0:
            break
        for p in range(0, N, n):
            for k in range(n_split):
                l = p + k
                x[l] ^= x[l + n_split]
        n = n_split

    return x


def build_generator_matrix(N):
    """构造 G_N = F^{\\otimes n}（自然序，无 B_N 行置换）。"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    while G.shape[0] < N:
        G = np.kron(G, F) % 2
    return G


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    print("u =", u, "-> x =", x, "G@u =", (u @ G) % 2)
