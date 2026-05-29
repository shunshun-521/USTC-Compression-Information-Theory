"""
极化码编码器
编码：x = u * G_N，G_N = B_N F^{\\otimes n}，蝶形 O(N log N)
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码：x = u * G_N，G_N = B_N F^{\\otimes n}。

    蝶形 XOR 后做比特倒序置换。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    for i in range(n):
        step = 2**i
        for j in range(0, N, 2 * step):
            for k in range(step):
                u[j + k] ^= u[j + k + step]
    return u[bit_reversal_permutation(N)]


def build_generator_matrix(N):
    """构造 G_N = B_N F^{\\otimes n}（用于验证）"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = np.array([[1]], dtype=int)
    for _ in range(int(np.log2(N))):
        G = np.kron(G, F) % 2
    br = bit_reversal_permutation(N)
    B = np.eye(N, dtype=int)[br]
    return (B @ G) % 2
