"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.arange(N, dtype=int)
    rev = np.zeros(N, dtype=int)
    for bit in range(n):
        rev += ((indices >> bit) & 1) << (n - 1 - bit)
    return rev


def bit_reversed_index(i, n):
    """单索引比特倒序"""
    result = 0
    for bit in range(n):
        if i & (1 << bit):
            result |= 1 << (n - 1 - bit)
    return result


def polar_encode_butterfly(u):
    """蝶形编码（不含比特倒序）。"""
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    block = N
    for _ in range(n):
        half = block // 2
        for start in range(0, N, block):
            for k in range(half):
                idx = start + k
                u[idx] ^= u[idx + half]
        block = half
    return u


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    x = butterfly(u) 再经 B_N 置换。
    """
    u = polar_encode_butterfly(u)
    brp = bit_reversal_permutation(len(u))
    return u[brp]


def polar_generator_matrix(N):
    """生成矩阵 G_N = B_N F^{\\otimes n}"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    while G.shape[0] < N:
        G = np.kron(G, F)
    brp = bit_reversal_permutation(N)
    return G[brp, :]


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix(4)
    x_mat = u @ G % 2
    print("butterfly+brp:", x, "matrix:", x_mat)
    assert np.array_equal(x, x_mat), f"编码器与矩阵不一致: {x} vs {x_mat}"

    u2 = np.array([1, 0, 1, 1])
    x2 = polar_encode_butterfly(u2)
    assert np.array_equal(x2, [1, 1, 0, 1]), f"蝶形编码错误: {x2}"
    print("Encoder tests passed")
