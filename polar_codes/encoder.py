"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        r, v = 0, i
        for _ in range(n):
            r = (r << 1) | (v & 1)
            v >>= 1
        rev[i] = r
    return rev


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    蝶形结构后做比特倒序，与 u @ B @ F^⊗n 等价。
    """
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    x = u.copy()
    n = int(np.log2(N))
    step = 1
    for _ in range(n):
        for i in range(0, N, 2 * step):
            x[i:i + step] ^= x[i + step:i + 2 * step]
        step *= 2
    br = bit_reversal_permutation(N)
    return x[br]


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u, "-> x =", x)
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"
    print("Encoder test passed.")
