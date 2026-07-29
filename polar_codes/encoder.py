"""
极化码编码器
编码：x = u * F^{\otimes n}，利用蝶形结构实现 O(N log N) 复杂度
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
    极化码编码（蝶形结构，O(N log N)）。
    编码完成后对码字做比特倒序置换，等价于 x = u * B_N * F^{\otimes n}。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    assert 2 ** n == N

    step = N
    while step > 1:
        half = step // 2
        for p in range(0, N, step):
            for k in range(half):
                u[p + k] ^= u[p + k + half]
        step = half

    rev = bit_reversal_permutation(N)
    return u[rev]


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    n = 2
    F = np.array([[1, 0], [1, 1]])
    G = F
    for _ in range(n - 1):
        G = np.kron(G, F)
    rev = bit_reversal_permutation(len(u))
    G_br = G[rev, :]
    x_expected = (u @ G_br) % 2
    assert np.array_equal(x, x_expected), f"编码器错误: {x} != {x_expected}"
    print("Encoder test passed:", x)
