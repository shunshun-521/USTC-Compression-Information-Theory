"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(format(i, f'0{n}b')[::-1], 2)
    return rev


def polar_encode(u):
    """
    极化码编码。

    利用蝶形结构计算 x = u * F^{\\otimes n}（O(N log N)）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    for i in range(n):
        step = 1 << i
        block = step << 1
        for j in range(N // 2):
            k = j // step
            offset = j % step
            a = k * block + offset
            b = a + step
            u[a] ^= u[b]
    return u


def polar_encode_with_bit_reversal(u):
    """带比特倒序置换的编码变体（供对比/测试）。"""
    x = polar_encode(u)
    rev = bit_reversal_permutation(len(x))
    return x[rev]


def polar_encode_matrix(u):
    """基于生成矩阵 G = F^{\\otimes n} 的编码（用于验证）"""
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    return (u @ G) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u)
    print("x =", x)
    assert np.array_equal(x, polar_encode_matrix(u))
    print("matrix match OK")
