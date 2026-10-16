r"""
极化码编码器
编码：x = u * F^{\otimes n}（蝶形结构，O(N log N)）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组：out[i] = in[perm[i]]"""
    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError("N must be a power of 2")
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def build_generator_matrix(N):
    r"""F^{\otimes n} (mod 2)，与蝶形编码一致"""
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=np.uint8)
    G = np.array([[1]], dtype=np.uint8)
    for _ in range(n):
        G = np.kron(G, F)
    return G % 2


def polar_encode(u):
    """
    极化码编码（含比特倒序置换，等价于 x = u @ G_N）。

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
            for j in range(step):
                a = i + j
                b = i + j + step
                x[a] ^= x[b]
        step <<= 1
    return x.astype(int)


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    x_mat = (u @ G) % 2
    print("butterfly+BR:", x)
    print("matrix:", x_mat)
