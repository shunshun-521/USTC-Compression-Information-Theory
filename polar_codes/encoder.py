"""
极化码编码器
编码：x = u * F^{\\otimes n}，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f'0{n}b')[::-1], 2) for i in range(N)])


def polar_encode(u):
    """
    极化码编码。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字，x = u * F^{\\otimes n}

    实现：蝶形（butterfly）递归结构
        - 每层：相邻对 (u[i], u[i + step]) -> (u[i] XOR u[i+step], u[i+step])
        - 共 log2(N) 层
    """
    u = np.array(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError("N must be a power of 2")

    for layer in range(n):
        step = 1 << layer
        for i in range(0, N, 2 * step):
            for j in range(step):
                u[i + j] ^= u[i + j + step]

    return u


def build_generator_matrix(N):
    """构建 F^{\\otimes n} 生成矩阵（用于验证）"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    while G.shape[0] < N:
        G = np.kron(G, F) % 2
    return G


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    x_ref = np.array([np.dot(u, G[:, j]) % 2 for j in range(4)])
    print("u:", u)
    print("x (butterfly):", x)
    print("x (matrix):", x_ref)
    assert np.array_equal(x, x_ref), "编码器与生成矩阵不一致"
