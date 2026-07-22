"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(bin(i)[2:].zfill(n)[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字

    实现：蝶形（butterfly）递归结构
        - 每层：u[i] <- u[i] XOR u[i + step]
        - 共 log2(N) 层
        - 最后做比特倒序置换（bit-reversal permutation）
    """
    u = np.asarray(u, dtype=np.int_)
    N = len(u)
    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError(f"N={N} must be a power of 2")

    x = u.copy()
    for stage in range(n):
        step = 2 ** stage
        for i in range(0, N, 2 * step):
            for j in range(i, i + step):
                x[j] = (x[j] + x[j + step]) % 2

    br = bit_reversal_permutation(N)
    return x[br]


def polar_generate_matrix(N):
    """生成 G_N = B_N F^{\\otimes n}（用于校验）"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int_)
    G = np.array([[1]], dtype=np.int_)
    for _ in range(int(np.log2(N))):
        G = np.kron(G, F)
    br = bit_reversal_permutation(N)
    return G[br, :] % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generate_matrix(4)
    x_mat = (u @ G) % 2
    print("butterfly:", x)
    print("matrix:   ", x_mat)
    assert np.array_equal(x, x_mat), "编码器与生成矩阵不一致"
