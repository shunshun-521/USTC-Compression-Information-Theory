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
    for i in range(n):
        rev |= ((indices >> i) & 1) << (n - 1 - i)
    return rev


def polar_encode(u):
    """
    极化码编码。

    蝶形结构：对 (u[i], u[i+step])，执行 u[i] ^= u[i+step]。
    译码器按比特倒序调度，与显式 B_N 置换等价。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    nblk = N
    for _ in range(int(np.log2(N))):
        n_split = nblk // 2
        for p in range(0, N, nblk):
            for k in range(n_split):
                l = p + k
                u[l] ^= u[l + n_split]
        nblk = n_split
    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u, "-> x =", x)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
    print("Encoder test passed.")
