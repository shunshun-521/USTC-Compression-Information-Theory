"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字

    实现：蝶形（butterfly）递归结构
        - 每层：相邻对 (u[i], u[i + step]) -> (u[i] XOR u[i+step], u[i+step])
        - 共 log2(N) 层
        - 最后做比特倒序置换（bit-reversal permutation）
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))

    for step in range(n):
        stride = 1 << step
        for i in range(0, N, 2 * stride):
            u[i : i + stride] ^= u[i + stride : i + 2 * stride]

    rev = bit_reversal_permutation(N)
    return u[rev]


if __name__ == "__main__":
    # 验证：与生成矩阵 G_N = B_N F^{⊗n} 一致
    N = 4
    F = np.array([[1, 0], [1, 1]])
    G = F
    for _ in range(int(np.log2(N)) - 1):
        G = np.kron(G, F)
    rev = bit_reversal_permutation(N)
    B = np.zeros((N, N), dtype=int)
    for i in range(N):
        B[i, rev[i]] = 1
    GN = (G @ B) % 2

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    x_mat = (u @ GN) % 2
    assert np.array_equal(x, x_mat), f"编码器与生成矩阵不一致: {x} vs {x_mat}"
    print("编码器校验通过: u =", u, "-> x =", x)
