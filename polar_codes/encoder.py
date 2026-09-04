"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=np.int64)
    rev = np.zeros(N, dtype=np.int64)
    for i in range(N):
        b = format(i, f"0{n}b")[::-1]
        rev[i] = int(b, 2)
    return rev


def polar_encode(u):
    """
    极化码编码（蝶形结构，O(N log N)）。

    与 permuted SC 译码器配套：蝶形 XOR 编码等价于 u @ F^{\\otimes n}。
    比特倒序由译码阶段的相位顺序隐式处理。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    assert 2 ** n == N

    for stage in range(n):
        step = 1 << stage
        for i in range(0, N, 2 * step):
            u[i:i + step] ^= u[i + step:i + 2 * step]

    return u.astype(int)


def build_generator_matrix(N):
    """构建 G_N = F^{\\otimes n}（用于验证）"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = F.copy()
    while G.shape[0] < N:
        G = np.kron(G, F)
    return G


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u:", u, "-> x:", x)
    G = build_generator_matrix(4)
    print("u @ G mod 2:", (u @ G) % 2)
