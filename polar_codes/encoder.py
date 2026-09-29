"""
极化码编码器
编码：u 经蝶形结构得到码字（与 Permuted SCD 译码配套，不在输出端做 B_N 置换）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.arange(N, dtype=np.int64)
    for i in range(N):
        r = 0
        v = i
        for _ in range(n):
            r = (r << 1) | (v & 1)
            v >>= 1
        rev[i] = r
    return rev


def polar_encode(u):
    """
    极化码编码，O(N log N) 蝶形 XOR（与 mcba1n/polar-codes 非递归编码一致）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = N
    while n > 1:
        n_split = n // 2
        for p in range(0, N, n):
            for k in range(n_split):
                l = p + k
                u[l] ^= u[l + n_split]
        n = n_split
    return u.astype(np.int8)


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("encode:", x)
    # Arikan G 与蝶形编码（无输出 BR）的参考码字
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
