"""
极化码编码器
编码：x = u * G_N，G_N = B_N F^{\\otimes n}；蝶形实现 F^{\\otimes n}，
比特倒序由 Permuted SCD 译码器侧等价处理（见 decoder_sc）。
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组：out[i] = bit_reverse(i)"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=np.int64)
    rev = np.zeros(N, dtype=np.int64)
    for b in range(n):
        rev |= ((idx >> b) & 1) << (n - 1 - b)
    return rev


def polar_encode(u):
    """
    极化码编码（蝶形 XOR，O(N log N)）。
    与 polarcodes / Permuted SCD 配套：不在此处对比特倒序，
    完整 G_N = B_N F^{\\otimes n} 由译码端置换抵消。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")

    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            u[i : i + step] ^= u[i + step : i + 2 * step]
        step <<= 1

    return u.astype(np.int8)


def polar_encode_with_bit_reversal(u):
    """显式比特倒序编码 x = (u F^{\\otimes n}) B_N（用于矩阵校验）。"""
    x = polar_encode(u)
    return x[bit_reversal_permutation(len(x))].astype(np.int8)


def polar_generator_matrix(N):
    """G_N = B_N F^{\\otimes n}（含比特倒序的完整生成矩阵）。"""
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = np.array([[1]], dtype=np.int8)
    for _ in range(n):
        G = np.kron(G, F)
    rev = bit_reversal_permutation(N)
    B = np.eye(N, dtype=np.int8)[rev]
    return (B @ G) % 2


def polar_generator_matrix_butterfly(N):
    """F^{\\otimes n}（与 polar_encode 直接输出一致）。"""
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = np.array([[1]], dtype=np.int8)
    for _ in range(n):
        G = np.kron(G, F)
    return G % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix_butterfly(4)
    print("butterfly:", x)
    print("matrix u@G:", (u.astype(np.int8) @ G) % 2)
    u2 = np.array([1, 0, 1, 1])
    x2 = polar_encode_with_bit_reversal(u2)
    Gfull = polar_generator_matrix(4)
    print("with rev:", x2, "u@Gfull", (u2 @ Gfull) % 2)
