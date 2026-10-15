"""
极化码编码器（Arikan F=[[1,1],[0,1]]，非递归 XOR 结构）
"""
import numpy as np


def bit_reversal_permutation(N):
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(format(i, f"0{n}b")[::-1], 2)
    return rev


def polar_encode(u):
    """
    极化码编码，与标准 Arikan 生成矩阵一致（O(N log N)）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    n = len(u)
    stage_len = n
    while stage_len > 1:
        half = stage_len // 2
        for p in range(0, n, stage_len):
            for k in range(half):
                l = p + k
                u[l] ^= u[l + half]
        stage_len = half
    return u % 2


def get_generator_matrix(N):
    """返回 GF(2) 生成矩阵（用于校验）"""
    n = int(np.log2(N))
    F = np.array([[1, 1], [0, 1]], dtype=np.int8)
    G = F
    for _ in range(n - 1):
        G = np.kron(G, F)
    return G.astype(np.int8) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = get_generator_matrix(4)
    assert np.array_equal(x, G @ u % 2)
    print("encoder test passed:", x)
