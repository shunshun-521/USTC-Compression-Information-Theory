"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（生成矩阵 G_N = F^{\\otimes n}，自然序）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    x = np.asarray(u, dtype=int).copy()
    N = len(x)
    n = int(np.log2(N))

    for s in range(n):
        step = 1 << s
        for i in range(0, N, 2 * step):
            for j in range(step):
                x[i + j] ^= x[i + j + step]

    return x


def polar_encode_with_reversal(u):
    """带输出比特倒序置换的编码（与部分教材定义一致）。"""
    x = polar_encode(u)
    br = bit_reversal_permutation(len(u))
    return x[br]
