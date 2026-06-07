"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(format(i, f"0{n}b")[::-1], 2)
    return rev


def bit_reversed_index(i, n):
    """单索引比特倒序"""
    return int(format(i, f"0{n}b")[::-1], 2)


def polar_encode(u):
    """
    极化码编码（蝶形结构，O(N log N)）。

    与 SC 译码器配套：对 u 执行蝶形 XOR，输出即为信道传输码字。
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

    return u


def build_generator_matrix(N):
    r"""构造 G_N = B_N * F^{\otimes n}（用于验证）。"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    n = int(np.log2(N))
    F_n = F.copy()
    for _ in range(n - 1):
        F_n = np.kron(F_n, F)
    rev = bit_reversal_permutation(N)
    return F_n[rev]


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    x_mat = (np.array(u) @ G) % 2
    print("u:", u)
    print("butterfly:", x)
    print("matrix:", x_mat)
    print("Encoder consistent:", np.array_equal(x, x_mat))
