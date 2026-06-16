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
        r = 0
        v = i
        for _ in range(n):
            r = (r << 1) | (v & 1)
            v >>= 1
        rev[i] = r
    return rev


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    实现 x = u @ B_N @ F^{\\otimes n}：
        1. 对 u 做蝶形 XOR（每层完整更新所有对）
        2. 对比特倒序后的位置输出码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    assert 2 ** n == N

    for layer in range(n):
        step = 1 << layer
        for i in range(0, N, 2 * step):
            for j in range(i, i + step):
                u[j] ^= u[j + step]

    br = bit_reversal_permutation(N)
    x = np.zeros(N, dtype=np.int8)
    for i in range(N):
        x[i] = u[br[i]]
    return x.astype(int)


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u, "-> x =", x)
