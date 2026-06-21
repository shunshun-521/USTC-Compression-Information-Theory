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


def inverse_bit_reversal_permutation(N):
    """返回比特倒序的逆置换索引"""
    br = bit_reversal_permutation(N)
    inv = np.zeros(N, dtype=int)
    inv[br] = np.arange(N)
    return inv


def polar_encode(u):
    """
    极化码编码（蝶形 XOR + 比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字，x = u * G_N，G_N = B_N * F^{\\otimes n}
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    assert 2 ** n == N

    step = N
    while step > 1:
        half = step // 2
        for start in range(0, N, step):
            for k in range(half):
                idx = start + k
                u[idx] ^= u[idx + half]
        step = half

    return u[bit_reversal_permutation(N)]


def polar_encode_matrix(u):
    """基于生成矩阵的编码（用于验证）"""
    u = np.asarray(u, dtype=int)
    N = len(u)
    n = int(np.log2(N))

    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = np.array([[1]], dtype=int)
    for _ in range(n):
        G = np.kron(G, F)

    br = bit_reversal_permutation(N)
    B = np.zeros((N, N), dtype=int)
    for i, j in enumerate(br):
        B[i, j] = 1

    G_N = (B @ G) % 2
    return (u @ G_N) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    x_mat = polar_encode_matrix(u)
    print("u:", u)
    print("butterfly encode:", x)
    print("matrix encode:", x_mat)
    assert np.array_equal(x, x_mat)

    for N in [8, 16, 32]:
        rng = np.random.default_rng(0)
        for _ in range(10):
            u = rng.integers(0, 2, N)
            assert np.array_equal(polar_encode(u), polar_encode_matrix(u))
    print("All encoder checks passed.")
