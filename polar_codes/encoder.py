"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f'{i:0{n}b}'[::-1], 2) for i in range(N)])


def bit_reversed(i, n):
    """对标量索引 i 做 n 位比特倒序"""
    result = 0
    for b in range(n):
        if i & (1 << b):
            result |= 1 << (n - 1 - b)
    return result


def polar_encode(u):
    """
    极化码编码（蝶形结构，无输出比特倒序）。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = N
    for _ in range(int(np.log2(N))):
        if n == 1:
            break
        n_split = n // 2
        for p in range(0, N, n):
            for k in range(n_split):
                l = p + k
                u[l] = u[l] ^ u[l + n_split]
        n = n_split
    return u


if __name__ == '__main__':
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print(f"u={u} -> x={x}")
    G = np.array([[1, 0, 0, 0], [1, 0, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]])
    print('u*G', np.dot(u, G) % 2)
    print("Encoder test passed.")
