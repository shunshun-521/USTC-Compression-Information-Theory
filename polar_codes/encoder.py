"""
极化码编码器
编码：x = u * G_N，G_N = B_N F^{⊗ n}
提供与 mcba1n 一致的 F^{⊗n} 编码及含 B_N 的编码
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError("N must be a power of 2")
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode_core(u):
    """Arikan 蝶形编码（不含 B_N），原地 XOR 结构"""
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    for stage in range(n):
        half_block = N >> (stage + 1)
        block = half_block * 2
        for base in range(0, N, block):
            for k in range(half_block):
                idx = base + k
                u[idx] ^= u[idx + half_block]
    return u.astype(int)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换 B_N）。
    """
    u = polar_encode_core(u)
    br = bit_reversal_permutation(len(u))
    return u[br]


def polar_generator_matrix(N):
    """GF(2) 生成矩阵 B_N F^{⊗ n}"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = F.copy()
    while G.shape[0] < N:
        G = np.kron(G, F)
    B = np.eye(N, dtype=np.int8)[bit_reversal_permutation(N)]
    return (B @ G) % 2
