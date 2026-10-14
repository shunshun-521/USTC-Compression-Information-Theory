"""
极化码编码器
编码：x = u F^{⊗ n}（蝶形 XOR，O(N log N)）
与 SC 译码器（McBain/Vangala 非递归实现）配套使用。
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError("N must be a power of 2")
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码：x = u F^{⊗ n}（非系统化，Arikan 核）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    x = u.copy()
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            for j in range(i, i + step):
                x[j] ^= x[j + step]
        step *= 2
    return x.astype(int)


def polar_decode_transform(x):
    """由码字 x 恢复源序列 u = x F^{⊗ n}（F 为自逆矩阵）"""
    return polar_encode(x)


def generator_matrix(N):
    """返回 F^{⊗ n}（用于校验）"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = np.array([[1]], dtype=int)
    n = int(np.log2(N))
    for _ in range(n):
        G = np.kron(G, F)
    return G


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = generator_matrix(4)
    x_ref = (G @ u) % 2
    print("u", u, "x", x, "ref", x_ref)
