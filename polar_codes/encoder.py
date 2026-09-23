"""
极化码编码器
O(N log N) 蝶形编码，与极化生成矩阵 G_N 等价
"""
import numpy as np

_GN_CACHE = {}


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def bit_reversed(x, n):
    """对标量索引 x 做 n 位比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def _build_GN(N):
    """通过穷举单位向量构建生成矩阵（仅用于校验，结果缓存）"""
    if N in _GN_CACHE:
        return _GN_CACHE[N]
    GN = np.zeros((N, N), dtype=np.int8)
    for j in range(N):
        u = np.zeros(N, dtype=np.int8)
        u[j] = 1
        GN[:, j] = polar_encode(u)
    _GN_CACHE[N] = GN
    return GN


def polar_encode(u):
    """
    极化码编码。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")

    n = int(np.log2(N)) + 1
    m = 1
    for _ in range(n - 1):
        for i in range(0, N, 2 * m):
            left = u[i : i + m].copy()
            right = u[i + m : i + 2 * m]
            u[i : i + m] = left ^ right
            u[i + m : i + 2 * m] = right
        m <<= 1
    return u


def polar_encode_matrix(u):
    """基于生成矩阵的编码，用于校验。"""
    u = np.asarray(u, dtype=np.int8)
    GN = _build_GN(len(u))
    return (GN @ u) % 2
