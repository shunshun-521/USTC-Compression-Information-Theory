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
        rev[i] = int(np.binary_repr(i, width=n)[::-1], 2)
    return rev


def polar_encode(u):
    """
    极化码编码（Kronecker 蝶形，大区块优先 XOR）。
    与 G_N = B_N F^{\\otimes n} 的标准极化码一致。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))

    for stage in range(1, n + 1):
        block = 1 << stage
        half = block // 2
        for kk in range(N // block):
            start = kk * block
            u[start:start + half] ^= u[start + half:start + block]

    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u, "-> x =", x)
