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
    极化码编码（蝶形 XOR，与译码器因子图一致）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    n = len(u)
    block = n
    while block > 1:
        half = block // 2
        for start in range(0, n, block):
            for k in range(half):
                idx = start + k
                u[idx] ^= u[idx + half]
        block //= 2
    return u


def build_generator_matrix(N):
    """构建 G_N = F^{\\otimes n}（无比特倒序，用于验证）"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = np.array([[1]], dtype=np.int8)
    n = int(np.log2(N))
    for _ in range(n):
        G = np.kron(G, F) % 2
    return G


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    x_mat = (u @ G) % 2
    print("butterfly:", x)
    print("matrix:", x_mat)
