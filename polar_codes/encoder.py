"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
（Arikan 蝶形，不在此处做额外比特倒序；与译码端约定一致）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（蝶形递归结构）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    x = np.array(u, dtype=np.int8, copy=True)
    N = len(x)
    n = int(np.log2(N))
    for stage in range(n):
        step = 2 ** (stage + 1)
        half = step // 2
        for i in range(0, N, step):
            for j in range(half):
                a = i + j
                b = i + j + half
                x[a] ^= x[b]
    return x.astype(int)


def polar_encode_with_bitrev(u):
    """带比特倒序置换的编码（G = B_N F^{⊗n} 标准形式）"""
    x = polar_encode(u)
    br = bit_reversal_permutation(len(u))
    return x[br]
