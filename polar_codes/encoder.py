"""
极化码编码器
编码：x = u * G_N（蝶形 / 分块 XOR），复杂度 O(N log N)
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.arange(N, dtype=int)
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        b = format(i, f"0{n}b")
        rev[i] = int(b[::-1], 2)
    return rev


def polar_encode(u):
    """
    极化码编码（Arikan 蝶形，与标准 SC/SCL 因子图一致）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位，冻结位应为 0）

    返回：
        x: 长度为 N 的码字（u 经极化变换后的结果）
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = N
    n_stages = int(np.log2(N))
    for _ in range(n_stages):
        if n == 1:
            break
        n_split = n // 2
        for p in range(0, N, n):
            for k in range(n_split):
                l = p + k
                u[l] ^= u[l + n_split]
        n = n_split
    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    print("polar_encode:", polar_encode(u))
