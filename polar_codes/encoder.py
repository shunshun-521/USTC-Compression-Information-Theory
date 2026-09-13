"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.arange(N, dtype=int)
    rev = ((indices[:, None] >> np.arange(n)) & 1).astype(int)
    rev = rev[:, ::-1]
    weights = 1 << np.arange(n)
    return (rev * weights).sum(axis=1)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=int).copy()
    n = int(np.log2(len(u)))
    step = 1
    for _ in range(n):
        for i in range(0, len(u), 2 * step):
            u[i:i + step] ^= u[i + step:i + 2 * step]
        step <<= 1

    br = bit_reversal_permutation(len(u))
    return u[br]


def polar_generator_matrix(N):
    """生成 G_N = B_N F^{\\otimes n}（用于验证）"""
    f = np.array([[1, 0], [1, 1]], dtype=int)
    g = np.array([[1]], dtype=int)
    for _ in range(int(np.log2(N))):
        g = np.kron(g, f)
    br = bit_reversal_permutation(N)
    return g[br, :]


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u)
    print("x =", x)
    g = polar_generator_matrix(4)
    print("G @ u =", (u @ g) % 2)
