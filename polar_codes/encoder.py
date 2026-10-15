"""
极化码编码器
编码：x = u * G_N，G_N = F^{\\otimes n}（与 SC/SCL 因子图一致）
利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=int)
    rev = np.zeros(N, dtype=int)
    for b in range(n):
        rev |= ((idx >> b) & 1) << (n - 1 - b)
    return rev


def polar_encode(u):
    """
    极化码编码（Arikan 蝶形，无额外比特倒序）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字，满足 x = u @ F^{\\otimes n} (mod 2)
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")
    n = int(np.log2(N))
    for s in range(n):
        step = 1 << (s + 1)
        half = 1 << s
        for i in range(0, N, step):
            for j in range(half):
                u[i + j] ^= u[i + j + half]
    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    print("encode:", polar_encode(u))
