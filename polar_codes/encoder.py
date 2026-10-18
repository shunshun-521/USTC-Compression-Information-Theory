"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    if 1 << n != N:
        raise ValueError("N must be a power of 2")
    return np.array(
        [int(("{:0%db}" % n).format(i)[::-1], 2) for i in range(N)],
        dtype=int,
    )


def polar_encode(u):
    """
    极化码编码（蝶形 + 比特倒序置换 B_N）。
    """
    x = np.asarray(u, dtype=np.int8).copy()
    N = len(x)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")
    n = N
    for _ in range(int(np.log2(N))):
        if n == 1:
            break
        n_split = n // 2
        for p in range(0, N, n):
            for k in range(n_split):
                x[p + k] ^= x[p + k + n_split]
        n = n_split
    br = bit_reversal_permutation(N)
    return x[br].astype(int)


def polar_generator_matrix(N):
    """生成 G_N = B_N F^{\\otimes n}（用于校验）"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    n = int(np.log2(N))
    for _ in range(n - 1):
        G = np.kron(G, F)
    br = bit_reversal_permutation(N)
    B = np.zeros((N, N), dtype=int)
    for i in range(N):
        B[br[i], i] = 1
    return (B @ G) % 2
