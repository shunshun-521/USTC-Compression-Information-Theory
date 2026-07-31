"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f'0{n}b')[::-1], 2) for i in range(N)], dtype=int)


def _butterfly_encode(u):
    """蝶形编码（不含比特倒序）。"""
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    block = N
    while block > 1:
        half = block // 2
        for base in range(0, N, block):
            for k in range(half):
                idx = base + k
                u[idx] ^= u[idx + half]
        block = half
    return u


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    """
    u = np.asarray(u, dtype=int)
    N = len(u)
    encoded = _butterfly_encode(u)
    return encoded[bit_reversal_permutation(N)]


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u)
    print("x =", x)
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F
    for _ in range(int(np.log2(len(u))) - 1):
        G = np.kron(G, F)
    G = G[:, bit_reversal_permutation(len(u))]
    x_ref = np.mod(u @ G, 2)
    assert np.array_equal(x, x_ref), f"编码器错误: {x} != {x_ref}"
    print("Encoder test passed.")
