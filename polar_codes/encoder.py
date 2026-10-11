"""
极化码编码器
编码：蝶形 XOR（与标准 polar_encode 一致，O(N log N)）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        r = 0
        for b in range(n):
            r = (r << 1) | ((i >> b) & 1)
        rev[i] = r
    return rev


def polar_encode(u):
    """
    极化码编码（蝶形结构，非系统化）。
    """
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    v = u.copy()
    stage_len = N
    while stage_len > 1:
        n_split = stage_len // 2
        for p in range(0, N, stage_len):
            for k in range(n_split):
                idx = p + k
                v[idx] ^= v[idx + n_split]
        stage_len = n_split
    return v.astype(int)


def generator_matrix(N):
    """G_N：对单位向量编码得到生成矩阵（用于校验）"""
    G = np.zeros((N, N), dtype=int)
    for i in range(N):
        e = np.zeros(N, dtype=int)
        e[i] = 1
        G[i] = polar_encode(e)
    return G
