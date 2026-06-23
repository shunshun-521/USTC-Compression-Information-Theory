"""
极化码编码器
编码：蝶形结构 O(N log N)，与 mcba1n 非递归编码一致
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(format(i, f"0{n}b")[::-1], 2)
    return rev


def bit_reversed(x, n):
    """对标量索引做比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def polar_generator_matrix(N):
    """构造生成矩阵 G_N = F^{⊗n}（与块级蝶形编码器一致）"""
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    A = np.array([[1]], dtype=int)
    for _ in range(n):
        A = np.kron(A, F)
    return A % 2


def polar_encode(u):
    """
    极化码编码（块级蝶形，从大块到小块 XOR）。
    信道传输码字为编码结果（不做输出比特倒序）。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    block = N
    for _ in range(int(np.log2(N))):
        if block == 1:
            break
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
