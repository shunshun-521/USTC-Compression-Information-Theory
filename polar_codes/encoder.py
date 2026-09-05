"""
极化码编码器
编码：x = u * F^{⊗ n}，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def bit_reversed_index(i, n):
    """单 index 的 bit-reversal（与 Vangala Permuted SCD 一致）"""
    result = 0
    for bit in range(n):
        if i & (1 << bit):
            result |= 1 << (n - 1 - bit)
    return result


def polar_encode(u):
    """
    极化码编码（Arikan 蝶形，无输出 bit-reversal）。

    与 Permuted SCD 译码器配套：x = u * F^{⊗ n}。
    """
    u = np.asarray(u, dtype=int).copy()
    n = int(np.log2(len(u)))
    if len(u) != 2 ** n:
        raise ValueError("Length of u must be a power of 2")

    stage_len = len(u)
    while stage_len > 1:
        half = stage_len // 2
        for block_start in range(0, len(u), stage_len):
            for k in range(half):
                idx = block_start + k
                u[idx] ^= u[idx + half]
        stage_len = half

    return u


def build_generator_matrix(N):
    """构造 G_N = F^{⊗ n}"""
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F) % 2
    return G
