"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组：out[i] = bit_reverse(i)"""
    n_bits = int(np.log2(N))
    return np.array([int(f"{i:0{n_bits}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    蝶形：u[i] ^= u[i+step]，最后 x[bit_rev(j)] = v[j]（与 G_N = B_N F^{⊗n} 一致）
    """
    u = np.asarray(u, dtype=np.int8).copy()
    n = len(u)
    step = 1
    while step < n:
        for i in range(0, n, 2 * step):
            for j in range(step):
                u[i + j] ^= u[i + j + step]
        step *= 2
    perm = bit_reversal_permutation(n)
    x = np.zeros(n, dtype=np.int8)
    x[perm] = u
    return x


def build_generator_matrix(N):
    """GF(2) 生成矩阵 F^{⊗log2(N)}（未含 B_N）"""
    F = np.array([[1, 1], [0, 1]], dtype=np.int8)
    G = np.array([[1]], dtype=np.int8)
    for _ in range(int(np.log2(N))):
        G = np.kron(G, F) % 2
    return G


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    perm = bit_reversal_permutation(4)
    xref = np.zeros(4, dtype=int)
    xref[perm] = (G @ u) % 2
    print("u=", u, "x=", x, "ref=", xref)
