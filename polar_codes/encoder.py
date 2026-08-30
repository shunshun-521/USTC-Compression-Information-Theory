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
        rev[i] = int(format(i, f"0{n}b")[::-1], 2)
    return rev


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

    for stage in range(n):
        step = 1 << stage
        for i in range(0, N, 2 * step):
            for j in range(step):
                u[i + j] ^= u[i + j + step]

    rev = bit_reversal_permutation(N)
    return u[rev]


def build_generator_matrix(N):
    """构建生成矩阵 G_N = B_N F^{⊗n}（用于验证）"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = F.copy()
    for _ in range(1, int(np.log2(N))):
        G = np.kron(G, F)
    rev = bit_reversal_permutation(N)
    G = G[rev, :]
    return G


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("polar_encode([1,0,1,1]) =", x)
    G = build_generator_matrix(4)
    x_ref = (u @ G) % 2
    print("matrix encode =", x_ref)
