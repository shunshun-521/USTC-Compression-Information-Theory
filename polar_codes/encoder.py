"""
极化码编码器
编码：u -> x，蝶形 XOR 结构，O(N log N)
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array(
        [int(format(i, f"0{n}b")[::-1], 2) for i in range(N)],
        dtype=int,
    )


def bit_reversed(x, n):
    """对标量或数组元素做比特倒序"""
    if np.isscalar(x):
        result = 0
        for i in range(n):
            if x & (1 << i):
                result |= 1 << (n - 1 - i)
        return result
    return np.array([bit_reversed(v, n) for v in np.asarray(x)], dtype=int)


def polar_encode(u):
    """
    极化码编码（蝶形 XOR，无输出比特倒序）。
    与译码器（按比特倒序信道索引 SC 译码）配套使用。
    """
    u = np.array(u, dtype=int).copy()
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
    return u


def build_generator_matrix(N):
    """构建生成矩阵 F^{\\otimes n}（无 B_N 行置换）"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(int(np.log2(N)) - 1):
        G = np.kron(G, F)
    return G


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    print("u =", u)
    print("x =", x)
    print("u @ G mod 2 =", (u @ G) % 2)
