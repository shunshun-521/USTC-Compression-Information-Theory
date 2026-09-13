"""
极化码编码器
编码：x = F^{\\otimes n} @ u（Arikan 标准核 [[1,1],[0,1]]）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组。"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=int)
    rev = ((idx[:, None] & (1 << np.arange(n))) != 0).astype(int)
    rev = rev[:, ::-1]
    weights = 1 << np.arange(n)
    return (rev * weights).sum(axis=1)


def polar_encode(u):
    """
    极化码 O(N log N) 编码（非系统化）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    n = len(u)
    stage = n
    while stage > 1:
        split = stage // 2
        for p in range(0, n, stage):
            for k in range(split):
                l = p + k
                u[l] ^= u[l + split]
        stage = split
    return u


def polar_encode_matrix(N):
    """生成 N x N 极化码生成矩阵 F^{\\otimes n}。"""
    F = np.array([[1, 1], [0, 1]], dtype=np.int8)
    G = F.copy()
    n = int(np.log2(N))
    for _ in range(n - 1):
        G = np.kron(F, G)
    return G


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_encode_matrix(4)
    print("u =", u)
    print("x =", x)
    print("G@u =", (G @ u) % 2)
