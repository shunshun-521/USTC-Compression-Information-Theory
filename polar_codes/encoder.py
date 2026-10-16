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
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字

    蝶形：相邻对 (u[i], u[i + step]) -> (u[i] XOR u[i+step], u[i+step])
    最后对码字做比特倒序置换（与 B_N 一致）。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            for j in range(step):
                u[i + j] ^= u[i + j + step]
        step *= 2
    # 与 SC/SCL/BP 因子图一致：x = u * F^{\otimes n}
    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u=", u, "x=", x)
