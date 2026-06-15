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
    极化码编码（3GPP / Arikan 蝶形结构，等价于 u @ G_N）。

    自顶向下（最大步长起）对每对子信道执行：
        (u_left, u_right) -> ((u_left + u_right) mod 2, u_right)

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    assert 2 ** n == N

    seq_len = 1
    for _ in range(n - 1, -1, -1):
        for i in range(0, N, 2 * seq_len):
            left = u[i : i + seq_len]
            right = u[i + seq_len : i + 2 * seq_len]
            u[i : i + seq_len] = (left + right) % 2
        seq_len *= 2
    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u)
    print("x =", x)
