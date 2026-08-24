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
    x = np.array(u, dtype=int).copy()
    n = len(x)
    if n == 0 or (n & (n - 1)) != 0:
        raise ValueError(f"Length {n} must be a power of 2")

    step = 1
    while step < n:
        for i in range(0, n, 2 * step):
            for j in range(step):
                x[i + j] ^= x[i + j + step]
        step *= 2

    br = bit_reversal_permutation(n)
    return x[br]


def build_generator_matrix(N):
    """构建 G_N = B_N F^{\\otimes n}（用于验证）。"""
    n = int(np.log2(N))
    f = np.array([[1, 0], [1, 1]], dtype=int)
    f_n = f.copy()
    for _ in range(n - 1):
        f_n = np.kron(f_n, f)
    br = bit_reversal_permutation(N)
    g = f_n[br, :]
    return g


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    g = build_generator_matrix(4)
    x_mat = np.dot(u, g) % 2
    print("butterfly encode:", x)
    print("matrix encode:  ", x_mat)
