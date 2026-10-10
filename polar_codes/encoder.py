"""
极化码编码器
编码：x = u * F^{⊗ n}（Arikan 核蝶形，O(N log N)）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（非系统化，与 F^{⊗ n} 矩阵乘法一致）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")
    step = N
    while step > 1:
        half = step // 2
        for base in range(0, N, step):
            for k in range(half):
                idx = base + k
                u[idx] ^= u[idx + half]
        step = half
    return u.astype(int)


def polar_encode_matrix(N):
    """生成矩阵 F^{⊗ n}，满足 x = (u @ G) % 2"""
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    Fn = np.array([[1]], dtype=int)
    for _ in range(n):
        Fn = np.kron(Fn, F)
    return Fn % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_encode_matrix(4)
    assert np.array_equal(x, (u @ G) % 2)
    print("u=", u, "x=", x)
