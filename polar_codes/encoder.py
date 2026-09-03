"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def _bit_reversed(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def polar_encode(u):
    """
    极化码编码（Arikan 蝶形：左半 ^= 右半）。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    block = N
    while block > 1:
        half = block // 2
        for p in range(0, N, block):
            for k in range(half):
                idx = p + k
                u[idx] = u[idx] ^ u[idx + half]
        block = half
    return u


if __name__ == "__main__":
    F = np.array([[1, 1], [0, 1]], dtype=int)
    G = F.copy()
    for _ in range(1):
        G = np.kron(G, F)
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, u @ G % 2), f"编码器错误: {x}"
    print("Encoder test passed:", x)
