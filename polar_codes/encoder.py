"""
极化码编码器
编码：x = G_N @ u (mod 2)，G_N = B_N F^{\\otimes n}
利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array(
        [int(format(i, f"0{n}b")[::-1], 2) for i in range(N)],
        dtype=int,
    )


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=int).copy()
    n = int(np.log2(len(u)))
    if len(u) != 2 ** n:
        raise ValueError("Length of u must be a power of 2")

    step = 1
    while step < len(u):
        for i in range(0, len(u), 2 * step):
            block = u[i : i + 2 * step]
            block[:step] ^= block[step:]
        step <<= 1

    br = bit_reversal_permutation(len(u))
    return u[br]


def polar_encode_matrix(u):
    """基于生成矩阵的编码，用于验证"""
    u = np.asarray(u, dtype=int)
    N = len(u)
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    F_n = F.copy()
    for _ in range(n - 1):
        F_n = np.kron(F_n, F)
    br = bit_reversal_permutation(N)
    B = np.eye(N, dtype=int)[br]
    G = B @ F_n
    return np.mod(G @ u, 2)


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u, "-> x =", x)
    assert np.array_equal(x, polar_encode_matrix(u))
