"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))

    for stage in range(n):
        step = 1 << (stage + 1)
        half = step >> 1
        for i in range(0, N, step):
            left = u[i:i + half]
            right = u[i + half:i + step]
            u[i:i + half] = left ^ right
            u[i + half:i + step] = right

    perm = bit_reversal_permutation(N)
    return u[perm]


def build_generator_matrix(N):
    """构造 G_N = B_N F^{⊗n}，用于验证"""
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    perm = bit_reversal_permutation(N)
    return G[perm, :]


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    x_ref = (u @ G) % 2
    print("butterfly:", x)
    print("matrix: ", x_ref)
    assert np.array_equal(x, x_ref), "编码器与生成矩阵不一致"
