"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f'0{n}b')[::-1], 2) for i in range(N)])


def polar_encode(u):
    """
    极化码编码。

    蝶形递归：对 (u[i], u[i+step]) 执行 u[i+step] ^= u[i]，
    等价于 x = u @ F^{⊗n}（自然序，无输出比特倒序）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    x = np.array(u, dtype=np.int8, copy=True)
    N = len(x)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")

    n = int(np.log2(N))
    for stage in range(n):
        step = 1 << stage
        for i in range(0, N, 2 * step):
            for j in range(i, i + step):
                x[j] ^= x[j + step]
    return x


def polar_encode_matrix(u):
    """基于生成矩阵的参考实现，用于校验。"""
    N = len(u)
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    Fn = F.copy()
    for _ in range(n - 1):
        Fn = np.kron(Fn, F)
    return (np.array(u, dtype=np.int8) @ Fn) % 2


if __name__ == '__main__':
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print('u =', u)
    print('x =', x)
    assert np.array_equal(x, polar_encode_matrix(u))
