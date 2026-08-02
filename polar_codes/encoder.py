"""
极化码编码器
编码：x = u * G_N，G_N = F^{⊗ n}（无比特倒序）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def build_generator_matrix(N):
    """构建生成矩阵 G_N = F^{⊗ n}"""
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    return G


def polar_encode(u):
    """
    极化码编码：x = u @ G_N mod 2
    """
    u = np.asarray(u, dtype=int)
    N = len(u)
    G = build_generator_matrix(N)
    return (u @ G) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    assert np.array_equal(x, u @ G % 2)
    print("Encoder test passed:", x)
