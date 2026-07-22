"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def build_generator_matrix(N):
    """构造 G_N = B_N F^{\\otimes n}（GF(2)）"""
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = np.array([[1]], dtype=np.int8)
    for _ in range(n):
        G = np.kron(G, F)
    br = bit_reversal_permutation(N)
    B = np.zeros((N, N), dtype=np.int8)
    for i, j in enumerate(br):
        B[i, j] = 1
    return (B @ G) % 2


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))

    for layer in range(n):
        step = 1 << layer
        for i in range(0, N, 2 * step):
            for j in range(step):
                u[i + j] ^= u[i + j + step]

    br = bit_reversal_permutation(N)
    return u[br].astype(int)


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    x_ref = (u @ G) % 2
    print("u:", u)
    print("x (butterfly):", x)
    print("x (matrix):", x_ref)
    assert np.array_equal(x, x_ref), "编码器与生成矩阵不一致"
