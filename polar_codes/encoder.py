"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
使用 Arikan 核 F = [[1,1],[0,1]] 的 Kronecker 积（无第三方库）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def bit_reversed(x, n):
    """单索引比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def _polar_encode_recursive(u, i1, i2):
    """递归极化编码（Arikan 核）"""
    h_shift = (i2 - i1 + 1) // 2
    if h_shift < 1:
        return
    mid = i1 + h_shift
    for k in range(i1, mid):
        u[k] ^= u[k + h_shift]
    if h_shift >= 2:
        _polar_encode_recursive(u, i1, mid - 1)
        _polar_encode_recursive(u, mid, i2)


def polar_encode(u):
    """
    极化码编码。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.array(u, dtype=int).copy()
    N = len(u)
    _polar_encode_recursive(u, 0, N - 1)
    return u


def polar_encode_generator_matrix(u):
    """通过生成矩阵验证编码（仅用于测试）"""
    N = len(u)
    n = int(np.log2(N))
    F = np.array([[1, 1], [0, 1]], dtype=int)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    return (G @ np.array(u, dtype=int)) % 2
