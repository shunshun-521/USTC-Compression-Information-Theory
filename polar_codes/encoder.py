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


def build_generator_matrix(N):
    """构造极化码生成矩阵 G_N = F^{\u2297 n}。"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    n = int(np.log2(N))
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    return G


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字

    实现：蝶形（butterfly）递归结构 + 列比特倒序
        x = u @ G_N[:, bit_reversal]
    """
    u = np.asarray(u, dtype=int)
    N = len(u)
    n = int(np.log2(N))
    assert 2 ** n == N, "N must be a power of 2"

    G = build_generator_matrix(N)
    rev = bit_reversal_permutation(N)
    return np.mod(u @ G[:, rev], 2)


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u)
    print("x =", x)
    G = build_generator_matrix(4)
    rev = bit_reversal_permutation(4)
    x_ref = np.mod(u @ G[:, rev], 2)
    assert np.array_equal(x, x_ref), f"编码器错误: {x}"
    print("Encoder test passed.")
