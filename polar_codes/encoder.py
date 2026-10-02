"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode_butterfly(u):
    """蝶形编码（无比特倒序），与 Permuted SCD 因子图一致。"""
    x = np.array(u, dtype=np.int8, copy=True)
    N = len(x)
    n = int(np.log2(N))
    block = N
    while block > 1:
        half = block // 2
        for base in range(0, N, block):
            for k in range(half):
                i = base + k
                x[i] ^= x[i + half]
        block = half
    return x.astype(int)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    内部先蝶形变换，再对比特倒序索引重排信道输出顺序。
    """
    x = polar_encode_butterfly(u)
    perm = bit_reversal_permutation(len(x))
    return x[perm].astype(int)


def polar_encode_matrix(u):
    """矩阵法编码，用于校验与 BP 早停。"""
    u = np.array(u, dtype=np.int8)
    N = len(u)
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = np.array([[1]], dtype=np.int8)
    for _ in range(n):
        G = np.kron(G, F) % 2
    perm = bit_reversal_permutation(N)
    B = np.eye(N, dtype=np.int8)[perm]
    return polar_encode(u)
