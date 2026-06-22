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

    蝶形结构：u[i] ^= u[i + step]，最后对比特倒序置换。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))

    block = N
    while block > 1:
        half = block // 2
        for start in range(0, N, block):
            for i in range(start, start + half):
                u[i] ^= u[i + half]
        block = half

    rev = bit_reversal_permutation(N)
    return u[rev]


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u)
    print("x =", x)

    # 与生成矩阵一致性验证
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(int(np.log2(len(u))) - 1):
        G = np.kron(G, F)
    P = np.zeros((len(u), len(u)), dtype=int)
    rev = bit_reversal_permutation(len(u))
    for i, j in enumerate(rev):
        P[i, j] = 1
    pg_u = (P @ G) @ u % 2
    assert np.array_equal(x, pg_u), f"编码器错误: {x} vs {pg_u}"
    print("Encoder generator-matrix test passed.")
