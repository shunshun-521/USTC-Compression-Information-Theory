"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.arange(N, dtype=int)
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        bits = format(i, f"0{n}b")[::-1]
        rev[i] = int(bits, 2)
    return rev


def polar_encode(u, apply_bit_reversal=False):
    """
    极化码编码：x = u * G_N，G_N = F^⊗n。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）
        apply_bit_reversal: 是否在输出端施加 B_N 比特倒序置换

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError(f"N={N} must be a power of 2")

    for layer in range(n):
        step = 1 << (layer + 1)
        half = step >> 1
        for start in range(0, N, step):
            for j in range(half):
                i = start + j
                u[i] = (u[i] + u[i + half]) % 2

    if apply_bit_reversal:
        u = u[bit_reversal_permutation(N)]
    return u


def polar_encode_matrix(u):
    """基于生成矩阵的编码（用于验证），G_N = F^⊗n。"""
    u = np.asarray(u, dtype=int)
    N = len(u)
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    return (u @ G) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u)
    print("x (butterfly) =", x)
    print("x (matrix)    =", polar_encode_matrix(u))
