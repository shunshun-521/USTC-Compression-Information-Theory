"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def bit_reversed(x, n):
    """对标量索引 x 做 n 位比特倒序。"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def polar_generator_matrix(N):
    """由单位向量编码结果构造 G_N（GF(2)）。"""
    N = int(N)
    G = np.zeros((N, N), dtype=int)
    for i in range(N):
        e = np.zeros(N, dtype=int)
        e[i] = 1
        G[i, :] = polar_encode(e)
    return G


def polar_encode(u):
    """
    极化码非递归编码（与生成矩阵乘法等价）。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    block = N
    for _ in range(N):
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
    G = polar_generator_matrix(4)
    x_ref = (u @ G) % 2
    print("u:", u)
    print("polar_encode:", x)
    print("matrix encode:", x_ref)
    assert np.array_equal(x, x_ref), "编码器与生成矩阵不一致"
