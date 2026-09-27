"""
极化码编码器
编码：u 上蝶形 XOR（与 polarcodes / Arikan 一致），x = polar_encode(u)
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    if 1 << n != N:
        raise ValueError("N must be a power of 2")
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def bit_reversed(i, n):
    return int(format(i, f"0{n}b")[::-1], 2)


def _polar_encode_core(u):
    """Arikan 蝶形编码（左半 ^= 右半）"""
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    for stage in range(1, n + 1):
        block = 1 << stage
        half = block // 2
        for base in range(0, N, block):
            u[base : base + half] ^= u[base + half : base + block]
    return u


def polar_encode(u):
    """
    极化码编码。
    为与 SC 译码树一致，采用蝶形 XOR 后做比特倒序（等效于 u @ G_N）。
    """
    x = _polar_encode_core(u)
    br = bit_reversal_permutation(len(x))
    return x[br]


def polar_encode_natural(u):
    """仅蝶形、不做倒序（供 BP / 对照）"""
    return _polar_encode_core(u)


def polar_generator_matrix(N):
    """G[j,:] = polar_encode(e_j)"""
    G = np.zeros((N, N), dtype=int)
    for j in range(N):
        ej = np.zeros(N, dtype=int)
        ej[j] = 1
        G[j, :] = polar_encode(ej)
    return G


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    G = polar_generator_matrix(4)
    print("u:", u)
    print("x:", polar_encode(u))
    print("u @ G:", (u @ G) % 2)
