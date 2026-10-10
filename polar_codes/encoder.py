"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
G_N = B_N F^(⊗n)，F = [[1,1],[0,1]]
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


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
    if 2**n != N:
        raise ValueError("N must be a power of 2")

    for level in range(n):
        step = 1 << level
        for i in range(0, N, 2 * step):
            for j in range(i, i + step):
                u[j] ^= u[j + step]

    perm = bit_reversal_permutation(N)
    return u[perm]


def polar_generator_matrix(N):
    r"""生成 G_N = B_N F^{\otimes n}（用于校验）"""
    F = np.array([[1, 1], [0, 1]], dtype=np.int8)
    G = np.array([[1]], dtype=np.int8)
    for _ in range(int(np.log2(N))):
        G = np.kron(G, F) % 2
    B = np.zeros((N, N), dtype=np.int8)
    perm = bit_reversal_permutation(N)
    for i, p in enumerate(perm):
        B[p, i] = 1
    return (B @ G) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix(4)
    x_ref = (u @ G) % 2
    print("u:", u, "x:", x, "matrix:", x_ref)
