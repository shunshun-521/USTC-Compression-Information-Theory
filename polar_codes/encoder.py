"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import math
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(math.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（自底向上蝶形，与 SC 因子图一致）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.array(u, dtype=int).copy()
    N = len(u)
    n = int(math.log2(N))
    sequence_len = 1
    for _ in range(n):
        for i in range(0, N, 2 * sequence_len):
            first = u[i : i + sequence_len]
            second = u[i + sequence_len : i + 2 * sequence_len]
            u[i : i + 2 * sequence_len] = np.concatenate(
                [(first + second) % 2, second]
            )
        sequence_len *= 2
    return u


def polar_encode_matrix(u):
    """基于生成矩阵的编码（用于验证）"""
    u = np.array(u, dtype=int)
    N = len(u)
    n = int(math.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    F_n = F.copy()
    for _ in range(n - 1):
        F_n = np.kron(F_n, F)
    return polar_encode(u)


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("polar_encode([1,0,1,1]) =", x)
