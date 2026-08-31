"""
极化码编码器
编码：蝶形（butterfly）结构，O(N log N)
与 Permuted SC 译码器配套（不在编码端做比特倒序）
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
    极化码编码（蝶形 XOR，无末尾比特倒序）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = N
    assert N > 0 and (N & (N - 1)) == 0, "N must be a power of 2"

    for _ in range(N):
        if n == 1:
            break
        half = n // 2
        for base in range(0, N, n):
            for k in range(half):
                u[base + k] ^= u[base + k + half]
        n = half

    return u


def build_generator_matrix(N):
    """构建标准极化码生成矩阵 G_N = B_N F^{\\otimes n}（供参考）"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(int(np.log2(N)) - 1):
        G = np.kron(G, F)
    br = bit_reversal_permutation(N)
    return G[br, :]


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("polar_encode([1,0,1,1]) =", x)
