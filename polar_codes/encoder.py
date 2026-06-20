"""
极化码编码器
编码：x = u * F^{⊗n}，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def bit_reversed(i, n):
    """单索引比特倒序"""
    result = 0
    for b in range(n):
        if i & (1 << b):
            result |= 1 << (n - 1 - b)
    return result


def polar_encode(u):
    """
    极化码编码（蝶形 XOR，与 G_N = F^{⊗n} 对应）。
    比特倒序置换由译码器在译码顺序中处理。
    """
    x = np.asarray(u, dtype=int).copy()
    N = len(x)
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            for j in range(step):
                a = i + j
                b = i + j + step
                x[a] ^= x[b]
        step <<= 1
    return x


def polar_encode_with_br(u):
    """含显式比特倒序的编码（u @ G_N = u @ F @ B）"""
    x = polar_encode(u)
    return x[bit_reversal_permutation(len(x))]


def build_f_matrix(N):
    """构建 F^{⊗n}（Arikan 生成矩阵，不含比特倒序）"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    F_n = F.copy()
    for _ in range(int(np.log2(N)) - 1):
        F_n = np.block([[F_n, np.zeros_like(F_n)], [F_n, F_n]])
    return F_n


def build_generator_matrix(N):
    """构建 G_N = B_N F^{⊗n}"""
    F_n = build_f_matrix(N)
    br = bit_reversal_permutation(N)
    return F_n[br, :]


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    x_br = polar_encode_with_br(u)
    print("u:", u)
    print("x (butterfly):", x)
    print("x (matrix G):", (u @ G) % 2)
    print("x (with BR):", x_br)
