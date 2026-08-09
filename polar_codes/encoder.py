"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.arange(N)
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(f"{i:0{n}b}"[::-1], 2)
    return rev


def build_generator_matrix(N):
    """构造极化码生成矩阵 G_N = B_N F^{\\otimes n}。"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    n = int(np.log2(N))
    for _ in range(n - 1):
        Z = np.zeros_like(G)
        G = np.block([[G, Z], [G, G]])
    rev = bit_reversal_permutation(N)
    return G[rev]


def gf2_inverse(A):
    """GF(2) 矩阵求逆。"""
    n = A.shape[0]
    A = A.copy()
    I = np.eye(n, dtype=int)
    for col in range(n):
        pivot = next((r for r in range(col, n) if A[r, col] == 1), None)
        if pivot is None:
            raise ValueError("Matrix not invertible in GF(2)")
        if pivot != col:
            A[[col, pivot]] = A[[pivot, col]]
            I[[col, pivot]] = I[[pivot, col]]
        for row in range(n):
            if row != col and A[row, col]:
                A[row] ^= A[col]
                I[row] ^= I[col]
    return I


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))

    for layer in range(n):
        step = 1 << layer
        for i in range(0, N, 2 * step):
            for j in range(step):
                u[i + j] ^= u[i + j + step]

    rev = bit_reversal_permutation(N)
    x = u[rev]
    return x.astype(int)


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u=", u, "x=", x)
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"
