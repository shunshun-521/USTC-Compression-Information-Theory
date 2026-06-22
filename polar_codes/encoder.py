"""
极化码编码器
编码：利用蝶形递归结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=np.int64)
    for i in range(N):
        b = format(i, f"0{n}b")[::-1]
        rev[i] = int(b, 2)
    return rev


def polar_encode(u):
    """
    极化码编码（非系统化，O(N log N) 蝶形结构）。

    每层将块长减半：u[l] ^= u[l + half]（左分区累加右分区，mod 2）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    n = int(np.log2(len(u)))
    block = len(u)
    for _ in range(n):
        half = block // 2
        for p in range(0, len(u), block):
            for k in range(half):
                l = p + k
                u[l] ^= u[l + half]
        block = half
    return u


def polar_encode_matrix(u):
    """基于生成矩阵 F^{\\otimes n}（Arikan 核 [[1,1],[0,1]]）的编码校验。"""
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    n = int(np.log2(N))
    F = np.array([[1, 1], [0, 1]], dtype=np.int8)
    G = np.array([[1]], dtype=np.int8)
    for _ in range(n):
        G = np.kron(G, F) % 2
    return (G @ u) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    xm = polar_encode_matrix(u)
    print("butterfly:", x)
    print("matrix:", xm)
    assert np.array_equal(x, xm)
