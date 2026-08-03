"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
G_N = B_N * F^(tensor n)
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f'0{n}b')[::-1], 2) for i in range(N)])


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字

    实现：蝶形（butterfly）递归结构
        - 每层：u[i] = (u[i] + u[i+step]) % 2（XOR 合并到左支）
        - 共 log2(N) 层
        - 最后做比特倒序置换（bit-reversal permutation）
    """
    u = np.array(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    step = 1
    for _ in range(n):
        for i in range(0, N, 2 * step):
            for j in range(step):
                u[i + j] = (u[i + j] + u[i + j + step]) % 2
        step *= 2
    rev = bit_reversal_permutation(N)
    return u[rev]


if __name__ == "__main__":
    # 编码器校验：与矩阵 G = B_N * F^{\otimes n} 一致
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    n = int(np.log2(len(u)))
    F = np.array([[1, 0], [1, 1]])
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    B = np.zeros((len(u), len(u)), dtype=int)
    for i in range(len(u)):
        j = int(format(i, f'0{n}b')[::-1], 2)
        B[i, j] = 1
    G_full = (B @ G) % 2
    x_mat = (u @ G_full) % 2
    assert np.array_equal(x, x_mat), f"编码器错误: {x} vs {x_mat}"
    print("编码器校验通过")
