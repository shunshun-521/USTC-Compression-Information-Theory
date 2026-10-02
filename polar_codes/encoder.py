"""
极化码编码器
编码：蝶形 XOR 结构 O(N log N)，与 Arikan 生成矩阵 F^{\\otimes n} 等价
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        b = format(i, f"0{n}b")
        rev[i] = int(b[::-1], 2)
    return rev


def bit_reversed_index(x, n):
    """单索引比特倒序（与 polarcodes.utils.bit_reversed 一致）"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def polar_encode(u):
    """
    极化码编码：对 u 做 in-place 蝶形 XOR（不额外乘 B_N）。
    信道发送的码字即为编码后的 u 向量。
    """
    u = np.array(u, dtype=int).copy()
    n = int(np.log2(len(u)))
    block = len(u)
    for _ in range(n):
        half = block // 2
        for base in range(0, len(u), block):
            for k in range(half):
                i = base + k
                u[i] ^= u[i + half]
        block = half
    return u


def polar_encode_matrix(u):
    """矩阵形式 x = u @ (B_N F^{\\otimes n})，仅用于校验"""
    u = np.array(u, dtype=int).ravel()
    N = len(u)
    G = np.array([[1]], dtype=int)
    nn = int(np.log2(N))
    for _ in range(nn):
        G = np.block([[G, np.zeros_like(G)], [G, G]]) % 2
    br = bit_reversal_permutation(N)
    G = G[br, :]
    return (u @ G) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("butterfly:", x)
    u2 = np.array([1, 0, 1, 1])
    assert np.array_equal(polar_encode(u2), polar_encode_matrix(u2))
    print("encoder OK")
