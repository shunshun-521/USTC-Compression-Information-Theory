"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f'{i:0{n}b}'[::-1], 2) for i in range(N)])


def _polar_transform(u):
    """计算 u @ F^{\otimes n}（蝶形结构）。"""
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    step = 1
    for _ in range(n):
        for i in range(0, N, 2 * step):
            for j in range(step):
                u[i + j] ^= u[i + j + step]
        step *= 2
    return u


def polar_encode(u):
    """
    极化码编码：x = u * B_N * F^{\otimes n}。
    蝶形计算 u * F^{\otimes n}，再按比特倒序置换输出。
    """
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    v = _polar_transform(u)
    br = bit_reversal_permutation(N)
  # x[j] = v[i] where br[i] = j  =>  x[br[i]] = v[i]
    x = np.empty(N, dtype=np.int8)
    x[br] = v
    return x
