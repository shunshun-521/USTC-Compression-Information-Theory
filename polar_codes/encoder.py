"""
极化码编码器
编码：x = u * F^{\\otimes n}，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def _bit_reversed(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([_bit_reversed(i, n) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（Arikan 蝶形结构，无输出比特倒序）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字，满足 x = u @ F^{\\otimes n} (mod 2)
    """
    x = np.asarray(u, dtype=np.int8).copy()
    n = int(np.log2(len(x)))
    for stage in range(n):
        step = 1 << stage
        for j in range(0, len(x), step << 1):
            for k in range(j, j + step):
                x[k] ^= x[k + step]
    return x


def build_generator_matrix(N):
    """构建 G_N = F^{\\otimes n}（用于验证）"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    while G.shape[0] < N:
        G = np.kron(G, F)
    return G % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    x_ref = (u @ G) % 2
    print("encoded:", x)
    print("matrix:", x_ref)
    assert np.array_equal(x, x_ref)
