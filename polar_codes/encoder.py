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
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    for s in range(n):
        step = 1 << s
        for i in range(0, N, 2 * step):
            for j in range(step):
                u[i + j] ^= u[i + j + step]
    return u


def polar_encode(u):
    """
    极化码编码。
    蝶形结构实现 u * F^{⊗n}；SC 译码器在比特倒序相位顺序下处理，信道 LLR 保持自然顺序。
    """
    return _butterfly_encode(u).astype(int)


def polar_encode_no_br(u):
    """与 polar_encode 相同（保留接口兼容性）。"""
    return polar_encode(u)
