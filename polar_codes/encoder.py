"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.arange(N)
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(format(i, f"0{n}b")[::-1], 2)
    return rev


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    蝶形结构：u[i] ^= u[i+step]，共 log2(N) 层，最后做比特倒序置换。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    step = N
    while step > 1:
        half = step // 2
        for block_start in range(0, N, step):
            for k in range(half):
                idx = block_start + k
                u[idx] ^= u[idx + half]
        step = half

    rev = bit_reversal_permutation(N)
    return u[rev]


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print(f"u={u} -> x={x}")
    # 编码-译码一致性由 SC 译码器单元测试验证
    print("Encoder module loaded.")
