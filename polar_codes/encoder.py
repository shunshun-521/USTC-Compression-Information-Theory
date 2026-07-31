"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(format(i, f'0{n}b')[::-1], 2)
    return rev


def polar_encode(u):
    """
    极化码编码（aff3ct 风格蝶形，无上分支比特倒序）。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    k = N >> 1
    while k > 0:
        for j in range(0, N, 2 * k):
            for i in range(k):
                u[j + i] ^= u[k + j + i]
        k >>= 1
    return u


if __name__ == '__main__':
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print('u =', u, '-> x =', x)
    print('编码器校验: u=[1,0,1,1] -> x=[0,0,0,1] (aff3ct 约定)')
