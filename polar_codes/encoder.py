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
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


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
    x = np.asarray(u, dtype=np.int8).copy()
    n = int(np.log2(len(x)))
    if 2 ** n != len(x):
        raise ValueError("Length of u must be a power of 2")

    for layer in range(n):
        step = 2 ** (layer + 1)
        half = step // 2
        for i in range(0, len(x), step):
            for j in range(half):
                x[i + j] ^= x[i + j + half]

    br = bit_reversal_permutation(len(x))
    return x[br]


def build_generator_matrix(N):
    """构造 G_N = B_N F^{\\otimes n}（用于验证）。"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    fn = F.copy()
    while fn.shape[0] < N:
        fn = np.kron(fn, F)
    br = bit_reversal_permutation(N)
    bn = np.eye(N, dtype=np.int8)[br]
    return (bn @ fn) % 2
