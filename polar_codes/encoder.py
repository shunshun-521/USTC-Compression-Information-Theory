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

    实现：蝶形（butterfly）递归结构
        - 每层：相邻对 (u[i], u[i + step]) -> (u[i] XOR u[i+step], u[i+step])
        - 共 log2(N) 层
        - 最后做比特倒序置换（bit-reversal permutation）
    """
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    n = int(np.log2(N))
    x = u.copy()
    for layer in range(n):
        step = 1 << (layer + 1)
        half = step >> 1
        for i in range(0, N, step):
            x[i:i + half] ^= x[i + half:i + step]
    br = bit_reversal_permutation(N)
    return x[br]


def polar_encode_matrix(u):
    """基于生成矩阵的编码，用于校验。"""
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    n = int(np.log2(N))
    f = np.array([[1, 0], [1, 1]], dtype=np.int8)
    g = f.copy()
    for _ in range(n - 1):
        g = np.kron(g, f)
    br = bit_reversal_permutation(N)
    b = np.zeros((N, N), dtype=np.int8)
    for i, j in enumerate(br):
        b[i, j] = 1
    gn = (b @ g) % 2
    return (u @ gn) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("polar_encode:", x)
    print("matrix encode:", polar_encode_matrix(u))
