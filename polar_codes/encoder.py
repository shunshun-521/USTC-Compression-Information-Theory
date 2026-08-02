"""
极化码编码器
编码：x = u * F^tensor n（Kronecker 变换）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(format(i, f'0{n}b')[::-1], 2)
    return rev


def polar_encode(u):
    """
    极化码编码（Kronecker 变换，与译码器匹配的非倒序约定）。
    x[start:start+step] ^= x[start+step:start+2*step]
    """
    x = np.asarray(u, dtype=int).copy()
    N = len(x)
    step = 1
    while step < N:
        for start in range(0, N, 2 * step):
            x[start:start + step] ^= x[start + step:start + 2 * step]
        step *= 2
    return x


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u, "-> x =", x)
    # u @ G mod 2，G 为 F^{\otimes 2}
    G = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]])
    assert np.array_equal(x, u @ G % 2), f"编码器错误: {x}"
    print("Encoder test passed.")
