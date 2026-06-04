r"""
极化码编码器
编码：x = u * G_N，G_N = B_N * F^{\otimes n}
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
    r"""
    极化码编码：蝶形 XOR 后做比特倒序置换。
    x = u * B_N * F^{\otimes n}
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    step = 1
    for _ in range(n):
        for i in range(0, N, 2 * step):
            for j in range(step):
                u[i + j] ^= u[i + j + step]
        step *= 2
    rev = bit_reversal_permutation(N)
    return u[rev]


def polar_encode_matrix(u):
    """矩阵形式编码（用于验证）"""
    N = len(u)
    n = int(np.log2(N))
    G = np.array([[1]], dtype=int)
    for _ in range(n):
        G = np.kron(G, [[1, 0], [1, 1]]) % 2
    rev = bit_reversal_permutation(N)
    B = np.eye(N, dtype=int)[rev]
    Gn = (B @ G) % 2
    return np.asarray(u, dtype=int) @ Gn % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    xm = polar_encode_matrix(u)
    print("u=", u, "x=", x, "matrix=", xm)
    assert np.array_equal(x, xm), f"编码器与矩阵形式不一致: {x} vs {xm}"
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"
