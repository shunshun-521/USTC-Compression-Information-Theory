"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.arange(N, dtype=int)
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(f"{i:0{n}b}"[::-1], 2)
    return rev


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字

    实现：蝶形（butterfly）递归结构
        - 从大到小块：u[l] ^= u[l + block/2]
        - 最后做比特倒序置换（bit-reversal permutation）
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    if N & (N - 1) != 0:
        raise ValueError("N must be a power of 2")

    n = N
    while n > 1:
        n_split = n // 2
        for p in range(0, N, n):
            for k in range(n_split):
                l = p + k
                u[l] ^= u[l + n_split]
        n = n_split

    rev = bit_reversal_permutation(N)
    x = u[rev]
    return x.astype(int)


if __name__ == "__main__":
    # N=4, u=[1,0,1,1] -> x=[1,0,1,1]（G = B_N F^{⊗n}）
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u, "-> x =", x)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}"

    # u=[0,0,1,1] -> [0,0,1,1]
    u2 = np.array([0, 0, 1, 1])
    x2 = polar_encode(u2)
    assert np.array_equal(x2, [0, 0, 1, 1]), f"编码器错误: {x2}"
    print("Encoder tests PASSED")
