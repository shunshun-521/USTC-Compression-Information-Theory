"""
极化码编码器
编码：x = u * G_N，G_N = F^{\\otimes n}，蝶形结构 O(N log N)
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组。"""
    n = int(np.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码：对自然序 u 执行蝶形变换，x = u * F^{\\otimes n}。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    step = N
    while step > 1:
        step //= 2
        for base in range(0, N, 2 * step):
            for k in range(step):
                u[base + k] ^= u[base + k + step]
    return u


def build_generator_matrix(N):
    """构造 Arikan 生成矩阵 F^{\\otimes n}。"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    n = int(np.log2(N))
    for _ in range(n - 1):
        G = np.kron(G, F)
    return G % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    x_mat = (u @ G) % 2
    print("butterfly:", x)
    print("matrix: ", x_mat)
    assert np.array_equal(x, x_mat), f"编码器与生成矩阵不一致: {x} vs {x_mat}"
    print("Encoder test passed (matches generator matrix).")
