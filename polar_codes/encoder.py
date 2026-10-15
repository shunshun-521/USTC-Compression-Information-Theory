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
    for b in range(n):
        rev |= ((idx >> b) & 1) << (n - 1 - b)
    return rev


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    assert 2**n == N

    for layer in range(n):
        step = 1 << layer
        span = step << 1
        for start in range(0, N, span):
            for j in range(start, start + step):
                u[j] ^= u[j + step]

    rev = bit_reversal_permutation(N)
    x = u[rev]
    return x.astype(np.int8)


def build_generator_matrix(N):
    """调试用：显式构造 G_N = B_N F^{\\otimes n}"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = F.copy()
    while G.shape[0] < N:
        G = np.kron(G, F)
    rev = bit_reversal_permutation(N)
    B = np.zeros((N, N), dtype=np.int8)
    for i, r in enumerate(rev):
        B[i, r] = 1
    return (B @ G) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("encode:", x)
    G = build_generator_matrix(4)
    x2 = (u @ G) % 2
    print("matrix:", x2)
