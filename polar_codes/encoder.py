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
        rev[i] = int(format(i, f"0{n}b")[::-1], 2)
    return rev


def polar_encode(u):
    """
    极化码编码。
    使用 Arikan 核 F=[[1,1],[0,1]] 的蝶形结构（与标准 SC 译码器配套）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    block = N
    for _ in range(n):
        half = block // 2
        for start in range(0, N, block):
            for j in range(half):
                idx = start + j
                u[idx] ^= u[idx + half]
        block = half
    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u, "-> x =", x)
    F = np.array([[1, 1], [0, 1]], dtype=int)
    Fn = np.array([[1]], dtype=int)
    for _ in range(2):
        Fn = np.kron(F, Fn)
    x_mat = u @ Fn % 2
    assert np.array_equal(x, x_mat), f"编码器错误: {x} vs {x_mat}"
    print("Encoder test passed.")
