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
    u = np.asarray(u, dtype=int).copy()
    n = int(np.log2(len(u)))
    for layer in range(n):
        step = 1 << layer
        for i in range(0, len(u), 2 * step):
            for j in range(i, i + step):
                u[j] ^= u[j + step]
    br = bit_reversal_permutation(len(u))
    return u[br]


def polar_encode_matrix(u):
    """基于生成矩阵 G = B_N F^{\\otimes n} 的 GF(2) 编码，用于校验。"""
    u = np.asarray(u, dtype=int)
    n = int(np.log2(len(u)))
    f = np.array([[1, 0], [1, 1]], dtype=int)
    f_n = f.copy()
    for _ in range(n - 1):
        f_n = np.kron(f_n, f)
    br = bit_reversal_permutation(len(u))
    b = np.zeros((len(u), len(u)), dtype=int)
    for i, j in enumerate(br):
        b[i, j] = 1
    g = np.mod(b @ f_n, 2)
    return np.mod(u @ g, 2)


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    xm = polar_encode_matrix(u)
    print("u =", u)
    print("polar_encode =", x)
    print("matrix encode =", xm)
    assert np.array_equal(x, xm)
