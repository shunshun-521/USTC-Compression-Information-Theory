"""
极化码编码器
编码：x = u * G_N，G_N = F^⊗n（蝶形结构，O(N log N)）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def bit_reversed(i, n):
    """单索引比特倒序"""
    result = 0
    for k in range(n):
        if i & (1 << k):
            result |= 1 << (n - 1 - k)
    return result


def polar_encode(u):
    """
    极化码编码（蝶形结构，与 G_N = F^{\otimes n} 等价）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    if 2**n != N:
        raise ValueError("Length of u must be a power of 2")

    block = N
    while block > 1:
        half = block // 2
        for base in range(0, N, block):
            for k in range(half):
                idx = base + k
                u[idx] ^= u[idx + half]
        block = half

    return u


def polar_encode_with_br(u):
    """编码后再做比特倒序置换（可选）"""
    x = polar_encode(u)
    return x[bit_reversal_permutation(len(u))]


def polar_encode_matrix(u):
    """矩阵法编码（用于校验）：x = u @ G_N mod 2"""
    u = np.asarray(u, dtype=int)
    N = len(u)
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F) % 2
    return (u @ G) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    print("encode:", polar_encode(u))
    print("matrix:", polar_encode_matrix(u))
