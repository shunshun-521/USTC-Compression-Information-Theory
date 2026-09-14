"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError("N must be a power of 2")
    return np.array(
        [int(format(i, f"0{n}b")[::-1], 2) for i in range(N)],
        dtype=np.int64,
    )


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字

    实现：蝶形（butterfly）递归结构
        - 每层：相邻对 (u[i], u[i + step]) -> (u[i] XOR u[i+step], u[i+step])
        - 共 log2(N) 层
        - 最后做比特倒序置换（bit-reversal permutation）
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    if N == 0 or (N & (N - 1)):
        raise ValueError("u length must be a positive power of 2")

    step = 1
    while step < N:
        for left in range(0, N, 2 * step):
            right = left + step
            u[left:right] ^= u[right : right + step]
        step <<= 1

    br = bit_reversal_permutation(N)
    return u[br].astype(np.int8)


def polar_encode_butterfly(u):
    """仅蝶形变换，不做比特倒序（用于内部校验）。"""
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    step = 1
    while step < N:
        for left in range(0, N, 2 * step):
            right = left + step
            u[left:right] ^= u[right : right + step]
        step <<= 1
    return u.astype(np.int8)


def polar_encode_matrix(u):
    """基于生成矩阵 G_N = B_N F^{⊗ n} 的编码（用于校验）。"""
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = np.array([[1]], dtype=np.int8)
    for _ in range(n):
        G = np.kron(G, F) % 2
    br = bit_reversal_permutation(N)
    B = np.zeros((N, N), dtype=np.int8)
    for i, j in enumerate(br):
        B[i, j] = 1
    GN = (G @ B) % 2
    return (u @ GN) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("butterfly+br encode:", x)
    print("matrix encode:", polar_encode_matrix(u))
