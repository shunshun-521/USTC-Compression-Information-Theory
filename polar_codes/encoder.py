"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
G_N = B_N F^{⊗ n}（比特倒序置换 × 极化核）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError("N must be a power of 2")
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError("N must be a power of 2")

    for stage in range(n):
        step = 1 << stage
        for i in range(0, N, 2 * step):
            for j in range(step):
                u[i + j] = (u[i + j] + u[i + j + step]) % 2

    br = bit_reversal_permutation(N)
    return u[br]


def polar_generator_matrix(N):
    """生成 G_N = B_N F^{⊗ n}，用于校验"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    n = int(np.log2(N))
    F_n = F.copy()
    for _ in range(n - 1):
        F_n = np.kron(F_n, F) % 2
    B = np.eye(N, dtype=int)
    br = bit_reversal_permutation(N)
    B = B[br]
    return (B @ F_n) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix(4)
    x_mat = (u @ G) % 2
    print("butterfly encode:", x)
    print("matrix encode:  ", x_mat)
    assert np.array_equal(x, x_mat)
