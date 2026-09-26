"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组 br，满足 out[i] = in[br[i]]"""
    n = int(np.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    u: 源序列；x: 码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    for layer in range(n):
        step = 1 << layer
        for i in range(0, N, step * 2):
            for j in range(i, i + step):
                u[j] ^= u[j + step]
    br = bit_reversal_permutation(N)
    x = u[br]
    return x.astype(int)


def polar_generator_matrix(N):
    """G_N = B_N F^{⊗ n}（GF(2)），用于校验"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    F_n = F.copy()
    for _ in range(int(np.log2(N)) - 1):
        F_n = np.kron(F_n, F)
    br = bit_reversal_permutation(N)
    G = F_n[br, :]
    return G


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix(4)
    x_g = (u @ G) % 2
    print("butterfly:", x, "matrix:", x_g)
