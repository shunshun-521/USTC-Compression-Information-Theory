r"""
极化码编码器
编码：x = u * G_N，G_N = B_N F^{\otimes n}，蝶形 O(N log N)
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组：out[i] = in[bitrev(i)]"""
    n = int(np.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    蝶形： (u[i], u[i+step]) -> (u[i] XOR u[i+step], u[i+step])
    最后 x[i] = v[bitrev(i)]
    """
    u = np.asarray(u, dtype=np.int8).copy()
    n = len(u)
    if n & (n - 1):
        raise ValueError("Length must be power of 2")

    step = 1
    while step < n:
        for i in range(0, n, 2 * step):
            for j in range(step):
                u[i + j] ^= u[i + j + step]
        step *= 2

    perm = bit_reversal_permutation(n)
    return u[perm].astype(int)


def generator_matrix(N):
    r"""返回 G_N = B_N F^{\otimes n}（用于校验）"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    m = int(np.log2(N))
    for _ in range(m - 1):
        G = np.kron(G, F) % 2
    perm = bit_reversal_permutation(N)
    B = np.eye(N, dtype=int)[perm]
    return (B @ G) % 2
