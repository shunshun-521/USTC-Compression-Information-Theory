"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
G_N = F^{⊗n}，F = [[1,1],[0,1]]（Arikan 标准核）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(format(i, f'0{n}b')[::-1], 2)
    return rev


def bit_reversed(x, n):
    """单索引比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def polar_encode(u):
    """
    极化码编码（非递归蝶形，与 G_N @ u 等价）。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    step = N // 2
    while step >= 1:
        for p in range(0, N, 2 * step):
            for k in range(step):
                u[p + k] ^= u[p + k + step]
        step //= 2
    return u


def polar_generator_matrix(N):
    """生成极化码生成矩阵 G_N = F^{⊗n}"""
    n = int(np.log2(N))
    F = np.array([[1, 1], [0, 1]], dtype=int)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(F, G)
    return G


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix(4)
    x_ref = G @ u % 2
    assert np.array_equal(x, x_ref), f"编码器错误: {x} != {x_ref}"
    print("Encoder test passed:", x)
