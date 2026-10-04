"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=np.int64)
    rev = np.zeros(N, dtype=np.int64)
    for i in range(N):
        r = 0
        v = i
        for _ in range(n):
            r = (r << 1) | (v & 1)
            v >>= 1
        rev[i] = r
    return rev


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    assert 2 ** n == N

    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            u[i : i + step] ^= u[i + step : i + 2 * step]
        step <<= 1

    br = bit_reversal_permutation(N)
    x = u[br]
    return x.astype(int)


def polar_decode_source(x):
    """
    从信道码字 x 恢复源向量 u（逆比特倒序 + 逆蝶形）。
    与 polar_encode 互逆。
    """
    x = np.asarray(x, dtype=np.int8).copy()
    N = len(x)
    n = int(np.log2(N))
    br = bit_reversal_permutation(N)
    v = x[br]
    step = N // 2
    while step >= 1:
        for i in range(0, N, 2 * step):
            v[i : i + step] ^= v[i + step : i + 2 * step]
        step //= 2
    return v.astype(int)


def polar_generator_matrix(N):
    """生成 G_N = B_N F^{\\otimes n}（用于校验）"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = F.copy()
    while G.shape[0] < N:
        G = np.kron(G, F)
    br = bit_reversal_permutation(N)
    G = G[br, :]
    return G % 2
