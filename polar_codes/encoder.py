"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def _butterfly_encode(u):
    """蝶形 XOR 编码（不含比特倒序）"""
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    stage_n = N
    for _ in range(n):
        n_split = stage_n // 2
        for p in range(0, N, stage_n):
            for k in range(n_split):
                idx = p + k
                u[idx] ^= u[idx + n_split]
        stage_n = n_split
    return u


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=int)
    N = len(u)
    if 2 ** int(np.log2(N)) != N:
        raise ValueError(f"N={N} must be a power of 2")
    x = _butterfly_encode(u)
    return x[bit_reversal_permutation(N)]


def butterfly_encode(u):
    """蝶形编码（不含输出比特倒序，供 BP 早停重编码使用）"""
    return _butterfly_encode(np.asarray(u, dtype=int))
