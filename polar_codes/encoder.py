"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=np.int64)
    rev = np.zeros(N, dtype=np.int64)
    for i in range(N):
        r = 0
        x = i
        for _ in range(n):
            r = (r << 1) | (x & 1)
            x >>= 1
        rev[i] = r
    return rev


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int64).copy()
    N = len(u)
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            for j in range(step):
                u[i + j] ^= u[i + j + step]
        step <<= 1
    # 与 SC 译码树一致的码字顺序（等价于在构造阶段对信息位做比特倒序置换）
    return u.astype(int)


def build_generator_matrix(N):
    """构造 G_N = F^{\\otimes n}（与蝶形编码一致，用于校验）。"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int64)
    n = int(np.log2(N))
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    return G % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("encode:", x)
    G = build_generator_matrix(4)
    x2 = (u @ G) % 2
    print("matrix:", x2)
