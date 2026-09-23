"""
极化码编码器
编码：x = F^{\\otimes n} u，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def _bit_reversal_indices(N):
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    return _bit_reversal_indices(N)


def arikan_generator(N):
    """构建 F^{\\otimes n}，Arikan 核 F=[[1,1],[0,1]]"""
    F = np.array([[1, 1], [0, 1]], dtype=int)
    G = F.copy()
    n = int(np.log2(N))
    for _ in range(n - 1):
        G = np.kron(F, G)
    return G


def polar_encode(u):
    """
    极化码编码（蝶形结构 + 比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.array(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    block = N
    for _ in range(n):
        half = block // 2
        for start in range(0, N, block):
            for k in range(half):
                idx = start + k
                u[idx] ^= u[idx + half]
        block = half

    brp = _bit_reversal_indices(N)
    return u[brp]


def build_generator_matrix(N):
    """构建与 polar_encode 一致的生成矩阵"""
    G = np.zeros((N, N), dtype=int)
    for k in range(N):
        u = np.zeros(N, dtype=int)
        u[k] = 1
        G[k] = polar_encode(u)
    return G


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("polar_encode([1,0,1,1]) =", x)
    G = build_generator_matrix(4)
    print("matrix encode =", (u @ G) % 2)
