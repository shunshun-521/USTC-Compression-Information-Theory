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
        rev[i] = int(format(i, f"0{n}b")[::-1], 2)
    return rev


def polar_encode(u):
    """
    极化码编码（Arikan 蝶形，不含 bit-reversal；与 SC 矩阵译码器配套）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")

    block = 1
    while block < N:
        for start in range(0, N, 2 * block):
            u[start:start + block] ^= u[start + block:start + 2 * block]
        block <<= 1

    return u


def build_generator_matrix(N):
    """构建 F^{⊗n} 生成矩阵（不含 B_N，与编码器一致）"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = F.copy()
    while G.shape[0] < N:
        G = np.block([[G, np.zeros((G.shape[0], G.shape[1]), dtype=np.int8)],
                      [G, G]])
    return G


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u)
    print("x =", x)
    G = build_generator_matrix(4)
    x_mat = (u @ G) % 2
    print("matrix encode:", x_mat)
