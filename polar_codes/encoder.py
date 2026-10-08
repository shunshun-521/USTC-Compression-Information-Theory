"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组：out[i] = codeword[bit_reverse(i)]"""
    n = int(np.log2(N))
    perm = np.arange(N, dtype=int)
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        r = 0
        v = i
        for _ in range(n):
            r = (r << 1) | (v & 1)
            v >>= 1
        rev[i] = r
    return rev


def _butterfly_xor(cw):
    """蝶形 XOR：u[i] ^= u[i+step]"""
    N = len(cw)
    n = int(np.log2(N))
    for stage in range(n):
        step = 1 << stage
        span = step << 1
        for base in range(0, N, span):
            for j in range(step):
                cw[base + j] ^= cw[base + j + step]


def build_generator_matrix(N):
    """G_N = B_N F^{\\otimes n}（用于校验）"""
    F = np.array([[1, 0], [1, 1]], dtype=np.uint8)
    G = np.array([[1]], dtype=np.uint8)
    while G.shape[0] < N:
        G = np.kron(G, F)
    perm = bit_reversal_permutation(N)
    return G[perm, :]


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.uint8).copy()
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")

    _butterfly_xor(u)
    perm = bit_reversal_permutation(N)
    x = u[perm]
    return x.astype(np.uint8)


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    x_mat = (u.astype(int) @ G) % 2
    print("butterfly+br:", x)
    print("matrix:", x_mat)
