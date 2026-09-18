"""
极化码编码器
编码：x = u * F_N^{\\otimes n}，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（蝶形结构，O(N log N)）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))

    for stage in range(1, n + 1):
        step = 1 << stage
        half = step // 2
        for block_start in range(0, N, step):
            for k in range(half):
                idx = block_start + k
                u[idx] = (u[idx] ^ u[idx + half]) % 2

    return u


def generate_matrix(N):
    """生成极化码生成矩阵 G_N = F^{\\otimes n}（用于验证）"""
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    return G


if __name__ == "__main__":
    N = 4
    G = generate_matrix(N)
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    x_mat = (u @ G) % 2
    print("u =", u)
    print("x (butterfly) =", x)
    print("x (matrix)    =", x_mat)
    assert np.array_equal(x, x_mat), f"编码器与生成矩阵不一致: {x} vs {x_mat}"
    print("编码器验证通过")
