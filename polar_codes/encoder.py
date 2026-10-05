"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
（与置换 SCD 配套：编码阶段不做额外比特倒序）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=np.int64)
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
    极化码编码（McBain 非递归实现，u[l] ^= u[l+n_split]）。
    """
    u = np.asarray(u, dtype=np.int64).copy()
    N = len(u)
    n = N
    while n > 1:
        n_split = n // 2
        for p in range(0, N, n):
            for k in range(n_split):
                l = p + k
                u[l] ^= u[l + n_split]
        n = n_split
    return u


def polar_encode_matrix(u):
    """通过显式生成矩阵编码（用于校验）"""
    N = len(u)
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=np.int64)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    u = np.asarray(u, dtype=np.int64)
    return (u @ G) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, polar_encode_matrix(u)), f"矩阵校验失败: {x}"
    u2 = np.array([0, 1, 0, 1])
    assert np.array_equal(polar_encode(u2), [0, 0, 1, 1]), polar_encode(u2)
    print("encoder OK")
