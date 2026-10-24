"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
（Arikan 蝶形，等价于 u @ F^{\otimes n}，不在输出端附加 B_N；
 译码端在 SC/SCL 中按比特倒序相位处理，与信道顺序一致。）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组：out[i] = in[perm[i]]"""
    n = int(np.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    x = np.asarray(u, dtype=np.int8).copy()
    n = len(x)
    m = int(np.log2(n))
    if 2**m != n:
        raise ValueError("Length of u must be a power of 2")

    for s in range(m):
        stride = 2 ** (s + 1)
        half = stride // 2
        for i in range(0, n, stride):
            for j in range(half):
                x[i + j] = (x[i + j] + x[i + j + half]) % 2

    return x


def polar_encode_matrix(u):
    """矩阵形式验证：u @ F^{\otimes n} mod 2"""
    u = np.asarray(u, dtype=int)
    N = len(u)
    m = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = np.array([[1]], dtype=int)
    for _ in range(m):
        G = np.kron(G, F)
    return (u @ G) % 2
