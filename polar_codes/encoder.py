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


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    x = u * B_N * F^{\\otimes n}
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    assert 2**n == N

    stride = N
    while stride > 1:
        half = stride // 2
        for start in range(0, N, stride):
            for i in range(start, start + half):
                u[i] ^= u[i + half]
        stride = half

    rev = bit_reversal_permutation(N)
    return u[rev]


def build_generator_matrix(N):
    """构造 G_N = B_N * F^{\\otimes n}（GF(2)）"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(int(np.log2(N)) - 1):
        G = np.kron(G, F)
    B = np.zeros((N, N), dtype=int)
    rev = bit_reversal_permutation(N)
    for i, j in enumerate(rev):
        B[i, j] = 1
    return (B @ G) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    x_mat = (u @ G) % 2
    print("u =", u)
    print("butterfly+br x =", x)
    print("matrix x =", x_mat)
    assert np.array_equal(x, x_mat), f"编码器与生成矩阵不一致: {x} vs {x_mat}"
    print("Encoder test passed.")
