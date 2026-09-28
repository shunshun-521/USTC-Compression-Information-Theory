"""
极化码编码器
编码：x = u * F^{\\otimes n}（蝶形 XOR），与 Vangala SC 译码器配套
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=np.int64)


def bit_reversed_index(i, n):
    """单索引比特倒序"""
    result = 0
    for k in range(n):
        if i & (1 << k):
            result |= 1 << (n - 1 - k)
    return result


def polar_encode(u):
    """
    极化码编码（蝶形结构，O(N log N)）。

    与 decoder_sc 中 Vangala 置换 SC 配套：码字 x 即为 u 经 F^{\\otimes n} 变换
    （与 mcba1n/polar-codes 非系统编码一致）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    n_len = u.shape[0]
    stage = n_len
    while stage > 1:
        half = stage // 2
        for base in range(0, n_len, stage):
            for k in range(half):
                left = base + k
                u[left] ^= u[left + half]
        stage = half
    return u


def polar_encode_with_brp(u):
    """含比特倒序置换的编码（G_N = B_N F^{\\otimes n}）"""
    x = polar_encode(u)
    return x[bit_reversal_permutation(len(x))]
