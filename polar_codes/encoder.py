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
    极化码编码：x = u * G_N，G_N = F^{\otimes n}（左支 XOR 蝶形，无额外比特倒序）。

    与 SC/SCL/BP 译码器的因子图约定一致。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    step = 1
    for _ in range(n):
        for i in range(0, N, 2 * step):
            for j in range(step):
                u[i + j] ^= u[i + j + step]
        step *= 2
    return u


def polar_encode_with_reversal(u):
    r"""含比特倒序置换的编码（用于与生成矩阵 G_N = B_N F^{\otimes n} 对照）。"""
    x = polar_encode(u)
    return x[bit_reversal_permutation(len(x))]


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u, "-> x =", x)
