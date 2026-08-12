"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.arange(N, dtype=np.int64)
    rev = np.zeros(N, dtype=np.int64)
    for b in range(n):
        rev = (rev << 1) | ((indices >> b) & 1)
    return rev


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    块级蝶形：从大块到小块逐级 XOR，最后对比特倒序索引输出。
    """
    u = np.asarray(u, dtype=np.int64).copy()
    N = len(u)
    n = int(np.log2(N))
    assert 2 ** n == N, "N must be a power of 2"

    block = N
    while block > 1:
        half = block // 2
        for start in range(0, N, block):
            u[start:start + half] ^= u[start + half:start + block]
        block //= 2

    br = bit_reversal_permutation(N)
    x = np.empty(N, dtype=np.int64)
    x[br] = u
    return x


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u, "-> x =", x)
    # G_N = B_N F^{⊗n}，u=[1,0,1,1] -> x=[1,0,1,1]
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"
    print("Encoder test passed.")
