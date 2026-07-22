"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f'0{n}b')[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码。

    蝶形结构：步长 N/2, N/4, ..., 1，每层 u[i] ^= u[i+step]。
  输出为 u * F^{\\otimes n}（与标准 SC 因子图一致，不做额外比特倒序）。
    """
    u = np.array(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    step = N // 2
    for _ in range(n):
        for i in range(0, N, 2 * step):
            for j in range(step):
                u[i + j] ^= u[i + j + step]
        step //= 2
    return u


if __name__ == '__main__':
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print('u =', u)
    print('x =', x)
    # 标准极化码编码结果（与 SC 因子图一致）
    assert np.array_equal(x, [1, 1, 0, 1]), f'编码器错误: {x}'
    print('编码器校验通过')
