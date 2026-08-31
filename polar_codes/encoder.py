"""
极化码编码器
编码：u * F^{⊗n}，F = [[1,1],[0,1]]，利用蝶形 XOR 结构 O(N log N)
与 Permuted SC 译码器配套使用（输出不做比特倒序置换）
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
    极化码编码（Arıkan 蝶形 XOR，无输出比特倒序）。

    每层对块大小 2^stage 执行：u[left] ^= u[right]
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    assert 2 ** n == N

    step = N
    while step > 1:
        half = step // 2
        for block in range(0, N, step):
            for i in range(half):
                u[block + i] ^= u[block + i + half]
        step = half
    return u


def polar_encode_matrix(u):
    """基于生成矩阵的编码（用于验证）"""
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    n = int(np.log2(N))
    F = np.array([[1, 1], [0, 1]], dtype=np.int8)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    return (u @ (G % 2)) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    x_mat = polar_encode_matrix(u)
    print("u =", u)
    print("x (butterfly) =", x)
    print("x (matrix)   =", x_mat)
    assert np.array_equal(x, x_mat), "butterfly vs matrix mismatch"
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
    print("Encoder test passed.")
