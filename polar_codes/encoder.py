"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.arange(N)
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int('{:0{}b}'.format(i, n)[::-1], 2)
    return rev


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    等价于 x = u * B_N * F^{\otimes n}（模 2）。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            u[i:i + step] = u[i:i + step] ^ u[i + step:i + 2 * step]
        step *= 2

    rev = bit_reversal_permutation(N)
    return u[rev]


if __name__ == '__main__':
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [0, 0, 1, 1]), f'编码器错误: {x}'
    print('Encoder test passed:', x)
