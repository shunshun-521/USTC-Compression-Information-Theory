"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.arange(N, dtype=np.int64)
    rev = np.zeros(N, dtype=np.int64)
    for i in range(N):
        rev[i] = int(format(i, f"0{n}b")[::-1], 2)
    return rev


def polar_encode(u):
    """
    极化码编码（蝶形结构，与 G_N = F^⊗n 一致）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    n = int(np.log2(len(u)))
    n_len = len(u)

    for stage in range(n):
        step = 1 << stage
        for base in range(0, n_len, step * 2):
            for j in range(base, base + step):
                u[j] ^= u[j + step]

    return u.astype(int)


def polar_encode_matrix(u):
    """矩阵乘法编码，用于校验。"""
    u = np.asarray(u, dtype=np.int8)
    n = int(np.log2(len(u)))
    f = np.array([[1, 0], [1, 1]], dtype=np.int8)
    g = f.copy()
    for _ in range(n - 1):
        g = np.kron(g, f)
    return (u @ g) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u, "-> x =", x)
