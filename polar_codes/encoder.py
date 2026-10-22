r"""
极化码编码器
编码：x = u * F^{\otimes n}（mod 2），蝶形 O(N log N)
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组：out[i] = in[bit_rev(i)]"""
    n = int(np.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（Arikan 非系统化形式，x = u F^⊗n mod 2）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位），0/1

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")

    n = int(np.log2(N))
    for layer in range(n):
        step = 1 << layer
        for i in range(0, N, 2 * step):
            for j in range(step):
                a = u[i + j]
                b = u[i + j + step]
                u[i + j] = (a + b) & 1
                u[i + j + step] = b

    return u.astype(int)


def polar_encode_matrix(u):
    r"""矩阵形式验证：x = u @ F^{\otimes n} mod 2"""
    u = np.asarray(u, dtype=int)
    N = len(u)
    n = int(np.log2(N))
    F2 = np.array([[1, 0], [1, 1]], dtype=int)
    F = F2.copy()
    for _ in range(n - 1):
        F = np.kron(F, F2)
    return (u @ F) % 2
