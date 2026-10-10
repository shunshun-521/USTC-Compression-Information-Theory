"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=np.int64)
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
    极化码编码：O(N log N) 蝶形结构，x = u * F^{⊗n}（与 SC 译码器配套）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")
    n = int(np.log2(N))
    for layer in range(1, n + 1):
        block = 1 << layer
        half = block // 2
        for start in range(0, N, block):
            u[start : start + half] ^= u[start + half : start + block]
    return u.astype(int)


def polar_encode_matrix(u):
    """矩阵法编码，用于校验蝶形实现。"""
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    n = int(np.log2(N))

    def f_power(n_):
        if n_ == 1:
            return np.array([[1, 0], [1, 1]], dtype=np.int8)
        f = f_power(n_ - 1)
        z = np.zeros_like(f)
        top = np.hstack([f, z])
        bot = np.hstack([f, f])
        return np.vstack([top, bot])

    F = f_power(n)
    br = bit_reversal_permutation(N)
    G = F[br, :]
    return (u @ G) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("polar_encode:", x)
    print("matrix:", polar_encode_matrix(u))
