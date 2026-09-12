"""
极化码编码器
编码：x = u * F^{\\otimes n}（蝶形结构，与置换 SC 译码器配套）
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
    极化码编码（蝶形 XOR，非系统性）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=int).copy()
    n = len(u)
    stage = n
    while stage > 1:
        n_split = stage // 2
        for p in range(0, n, stage):
            for k in range(n_split):
                u[p + k] ^= u[p + k + n_split]
        stage = n_split
    return u


def build_generator_matrix(N):
    """构建生成矩阵 G_N = F^{\\otimes n}（用于验证）"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    while G.shape[0] < N:
        G = np.kron(G, F)
    return G


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("polar_encode([1,0,1,1]) =", x)
    G = build_generator_matrix(4)
    x_mat = (u @ G) % 2
    print("matrix encode =", x_mat)
    assert np.array_equal(x, x_mat), "编码器与生成矩阵不一致"
