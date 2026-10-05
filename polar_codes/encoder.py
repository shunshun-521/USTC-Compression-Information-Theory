"""
极化码编码器
编码：x = u · F^{⊗n}（与标准 polar-codes 库一致）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=int)
    bits = ((idx[:, None] >> np.arange(n)) & 1)[:, ::-1]
    return (bits * (2 ** np.arange(n))).sum(axis=1).astype(int)


def bit_reversed(i, n):
    result = 0
    for bit in range(n):
        if i & (1 << bit):
            result |= 1 << (n - 1 - bit)
    return result


def polar_encode(u):
    """非递归极化编码，O(N log N)。"""
    u = np.array(u, dtype=np.int8).copy()
    N = len(u)
    n = N
    while n > 1:
        n_split = n // 2
        for p in range(0, N, n):
            for k in range(n_split):
                l = p + k
                u[l] ^= u[l + n_split]
        n = n_split
    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    F = np.array([[1, 1], [0, 1]], dtype=np.int8)
    G = np.kron(F, F)
    assert np.array_equal(x, (u @ G) % 2), f"编码器错误: {x}"
    print("encoder test passed:", x)
