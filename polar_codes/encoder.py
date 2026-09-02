"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int_)
    N = len(u)
    n = int(np.log2(N))
    v = u.copy()
    step = 1
    for _ in range(n):
        for i in range(0, N, 2 * step):
            v[i : i + step] ^= v[i + step : i + 2 * step]
        step *= 2
    brp = bit_reversal_permutation(N)
    return v[brp]


def polar_encode_matrix(u):
    """基于生成矩阵的编码（用于验证）。"""
    u = np.asarray(u, dtype=np.int_)
    N = len(u)
    n = int(np.log2(N))
    f = np.array([[1, 0], [1, 1]], dtype=np.int_)
    fn = f.copy()
    for _ in range(n - 1):
        fn = np.kron(fn, f)
    brp = bit_reversal_permutation(N)
    bn = np.eye(N, dtype=np.int_)
    bn = bn[brp]
    g = (bn @ fn) % 2
    return (u @ g) % 2
