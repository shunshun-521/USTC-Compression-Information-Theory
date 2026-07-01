"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    蝶形结构：u[l] ^= u[l + block]（将右半区累加到左半区，mod 2）
    最后对比特倒序索引做置换。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    assert 2 ** n == N

    block = N
    while block > 1:
        half = block // 2
        for start in range(0, N, block):
            for k in range(half):
                l = start + k
                u[l] ^= u[l + half]
        block = half

    brp = bit_reversal_permutation(N)
    return u[brp]


def build_generator_matrix(N):
    """构造 G_N = B_N F^{\\otimes n}（用于校验）。"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F
    for _ in range(int(np.log2(N)) - 1):
        G = np.kron(G, F)
    B = np.eye(N, dtype=int)[bit_reversal_permutation(N)]
    return (B @ G) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    x_mat = u @ G % 2
    print("u:", u, "-> x:", x, "(matrix:", x_mat, ")")
    assert np.array_equal(x, x_mat), f"编码器与生成矩阵不一致: {x} vs {x_mat}"
    print("Encoder test passed.")
