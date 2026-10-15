"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组：out[i] = i 的比特倒序"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=np.int64)
    rev = np.zeros(N, dtype=np.int64)
    for i in range(N):
        r = 0
        v = i
        for _ in range(n):
            r = (r << 1) | (v & 1)
            v >>= 1
        rev[i] = r
    return rev


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    """
    x = np.array(u, dtype=np.int8, copy=True)
    N = len(x)
    n = int(np.log2(N))
    for stage in range(n):
        step = 1 << stage
        for i in range(0, N, 2 * step):
            for j in range(i, i + step):
                x[j] ^= x[j + step]
    br = bit_reversal_permutation(N)
    return x[br].astype(int)


def generator_matrix(N):
    """G_N = B_N F^{\\otimes n}（用于校验）"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    n = int(np.log2(N))
    G = np.array([[1]], dtype=np.int8)
    for _ in range(n):
        G = np.kron(G, F)
    br = bit_reversal_permutation(N)
    return G[br, :] % 2


def polar_encode_matrix(u):
    """矩阵乘法编码（与 G_N 严格一致）"""
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    G = generator_matrix(N)
    return (u @ G) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = generator_matrix(4)
    x_ref = (u @ G) % 2
    print("butterfly+BR:", x)
    print("matrix mult:", x_ref)
