"""
极化码编码器
编码：x = u * G_N，G_N = B_N F^{⊗ n}，蝶形 O(N log N)
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def build_generator_matrix(N):
    """构造 G_N = B_N F^{⊗ n}（行向量编码 u @ G_N）"""
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    F_n = np.array([[1]], dtype=int)
    for _ in range(n):
        F_n = np.kron(F_n, F) % 2
    br = bit_reversal_permutation(N)
    B = np.zeros((N, N), dtype=int)
    for i, j in enumerate(br):
        B[i, j] = 1
    return (B @ F_n) % 2


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    蝶形：x[j] ^= x[j+step]，最后对码字做比特倒序。
    """
    x = np.asarray(u, dtype=np.int8).copy()
    N = len(x)
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            for j in range(i, i + step):
                x[j] ^= x[j + step]
        step <<= 1
    br = bit_reversal_permutation(N)
    return x[br].astype(int)


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    x_ref = (u @ G) % 2
    print("u:", u)
    print("x (butterfly):", x)
    print("x (matrix):", x_ref)
    assert np.array_equal(x, x_ref)
