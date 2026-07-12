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
    for k in range(n):
        if i & (1 << k):
            result |= 1 << (n - 1 - k)
    return result


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    蝶形：分块 XOR，等价于 u @ (B_N F^{⊗n})。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    block = N
    while block > 1:
        half = block // 2
        for p in range(0, N, block):
            for k in range(half):
                u[p + k] ^= u[p + k + half]
        block = half
    return u[bit_reversal_permutation(N)]


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F
    for _ in range(int(np.log2(len(u))) - 1):
        G = np.kron(G, F)
    br = bit_reversal_permutation(len(u))
    Gn = np.mod(np.eye(len(u), dtype=int)[br] @ G, 2)
    x_ref = np.mod(u @ Gn, 2)
    print("u =", u, "-> x =", x, "(ref =", x_ref, ")")
    assert np.array_equal(x, x_ref), f"编码器错误: {x}"
    print("Encoder test passed.")
