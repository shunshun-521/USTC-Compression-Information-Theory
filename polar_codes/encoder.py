"""
极化码编码器
编码：x = u * G_N，G_N = B_N F^{⊗ n}，O(N log N) 蝶形实现
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组：out[i] = bit_reverse(i)"""
    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError("N must be a power of 2")
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def _f_kron_power(n):
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F) % 2
    return G


def polar_encode(u):
    """
    极化码编码：x = u * G_N，G_N = B_N F^{⊗ n}（蝶形 + 比特倒序）。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError("N must be a power of 2")

    for layer in range(n):
        step = 2 ** layer
        for i in range(0, N, 2 * step):
            for j in range(i, i + step):
                u[j] ^= u[j + step]

    br = bit_reversal_permutation(N)
    x = np.zeros(N, dtype=int)
    x[br] = u
    return x


def build_generator_matrix(N):
    """构造 G_N = B_N F^{⊗ n}（与 polar_encode 一致）"""
    n = int(np.log2(N))
    G = _f_kron_power(n)
    br = bit_reversal_permutation(N)
    B = np.zeros((N, N), dtype=int)
    for i in range(N):
        B[i, br[i]] = 1
    return (B @ G) % 2
