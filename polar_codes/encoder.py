"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.arange(N)[:, None]
    bits = (indices >> np.arange(n - 1, -1, -1)) & 1
    return bits.sum(axis=1)


def polar_encode(u):
    """
    极化码编码（蝶形递归结构，O(N log N)）。

    采用分层蝶形 XOR，与 SC/SCL/BP 译码器因子图一致。
    """
    u = np.asarray(u, dtype=np.int64).copy()
    N = len(u)
    n = int(np.log2(N))
    assert 2 ** n == N

    block = N
    while block > 1:
        half = block // 2
        for start in range(0, N, block):
            for k in range(half):
                idx = start + k
                u[idx] ^= u[idx + half]
        block = half

    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u)
    print("x =", x)

    F = np.array([[1, 1], [0, 1]], dtype=int)
    G = np.array([[1]], dtype=int)
    while G.shape[0] < 4:
        G = np.kron(G, F)
    assert np.array_equal(x, u @ G % 2), f"编码器错误: {x}"
    print("编码器校验通过")
