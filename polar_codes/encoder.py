"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import math
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(math.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(format(i, f'0{n}b')[::-1], 2)
    return rev


def polar_encode(u):
    """
    极化码编码（Arikan 蝶形结构，自下而上合并）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    tree_depth = int(math.log2(N))
    sequence_len = 1

    for _ in range(tree_depth - 1, -1, -1):
        for i in range(0, N, 2 * sequence_len):
            first = u[i:i + sequence_len]
            second = u[i + sequence_len:i + 2 * sequence_len]
            u[i:i + 2 * sequence_len] = np.concatenate([(first + second) % 2, second])
        sequence_len *= 2

    return u


if __name__ == "__main__":
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = np.kron(F, F) % 2
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = (u @ G) % 2
    print("u =", u)
    print("x =", x)
    print("expected (u@G) =", expected)
    assert np.array_equal(x, expected), f"编码器错误: {x}"
    print("Encoder test passed.")
