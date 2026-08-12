"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.arange(N, dtype=np.int64)
    if n == 0:
        return indices
    bits = ((indices[:, None] >> np.arange(n - 1, -1, -1)) & 1)
    rev = (bits[:, ::-1] * (1 << np.arange(n))).sum(axis=1)
    return rev.astype(np.int64)


def polar_encode(u):
    """
    极化码编码（Arikan 蝶形结构，无额外比特倒序）。
    """
    u = np.array(u, dtype=np.int8, copy=True)
    N = len(u)
    n = int(np.log2(N))
    block = N

    for _ in range(n):
        half = block // 2
        for start in range(0, N, block):
            for k in range(half):
                idx = start + k
                u[idx] ^= u[idx + half]
        block = half

    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
    print("Encoder test passed:", x)
