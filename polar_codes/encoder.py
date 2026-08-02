"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f'{i:0{n}b}'[::-1], 2) for i in range(N)])


def polar_encode(u):
    """
    极化码编码。

    蝶形结构：从大块到小块逐层 XOR（等价于 u * F^⊗n）。
    译码器在比特倒序索引顺序下处理，与编码约定一致。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = N
    while n > 1:
        n_split = n // 2
        for p in range(0, N, n):
            for k in range(n_split):
                u[p + k] ^= u[p + k + n_split]
        n = n_split
    return u


if __name__ == '__main__':
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print(f'u={u} -> x={x}')
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f'编码器错误: {x}'
    print('编码器校验通过')
