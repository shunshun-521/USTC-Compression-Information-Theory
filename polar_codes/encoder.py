r"""
极化码编码器
编码：x = u * F^{\otimes n}，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def bit_reversed(i, n):
    """单索引比特倒序。"""
    result = 0
    for b in range(n):
        if i & (1 << b):
            result |= 1 << (n - 1 - b)
    return result


def build_generator_matrix(N):
    r"""构造 Arikan 生成矩阵 F^{\otimes n}，F=[[1,1],[0,1]]。"""
    F = np.array([[1, 1], [0, 1]], dtype=int)
    G = F.copy()
    n = int(np.log2(N))
    for _ in range(n - 1):
        G = np.kron(G, F)
    return G % 2


def polar_encode(u):
    """
    极化码编码（非系统化）：x = u * G_N。

    蝶形：从大块到小块，上支路累加下支路（mod 2）。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    if N <= 0 or (N & (N - 1)) != 0:
        raise ValueError("u length must be a power of 2")

    block = N
    while block > 1:
        half = block // 2
        for base in range(0, N, block):
            for k in range(half):
                u[base + k] = (u[base + k] + u[base + k + half]) % 2
        block = half
    return u


def polar_decode_codeword(x):
    """从码字 x 恢复源序列 u = x * G_N（GF(2) 下 G 为自逆矩阵结构）。"""
    return polar_encode(x)


def polar_encode_partial(u_left):
    """对左子树已译比特做部分编码，供 SC/SCL 的 g 运算使用。"""
    return polar_encode(u_left)
