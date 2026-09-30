"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组：out[i] = bit_reverse(i)"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=np.int64)
    rev = np.zeros(N, dtype=np.int64)
    for b in range(n):
        rev |= ((idx >> b) & 1) << (n - 1 - b)
    return rev


def polar_encode(u):
    """
    极化码编码：分阶段蝶形（与 Permuted SCD 配套）
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = N
    while n > 1:
        half = n // 2
        for base in range(0, N, n):
            for k in range(half):
                u[base + k] ^= u[base + k + half]
        n = half
    return u.astype(np.int8)


def polar_encode_no_br(u):
    """与 polar_encode 相同（保留接口供 BP 早停）"""
    return polar_encode(u)
