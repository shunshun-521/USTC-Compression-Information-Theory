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
        rev[i] = int("".join(reversed(format(i, f"0{n}b"))), 2)
    return rev


def bit_reversed_index(i, n):
    """对标量索引 i 做比特倒序"""
    result = 0
    for bit in range(n):
        if i & (1 << bit):
            result |= 1 << (n - 1 - bit)
    return result


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字

    实现：蝶形结构
        - 每层：u[i] ^= u[i + step]（左支累加右支，mod 2）
        - 共 log2(N) 层
        - 最后做比特倒序置换
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    assert 2 ** n == N

    block = N
    while block > 1:
        half = block // 2
        for start in range(0, N, block):
            for j in range(half):
                idx = start + j
                u[idx] ^= u[idx + half]
        block = half

    rev = bit_reversal_permutation(N)
    return u[rev]


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u)
    print("x =", x)
    # 与生成矩阵 u @ B @ G 一致（标准极化码）
    F = np.array([[1, 0], [1, 1]])
    G = np.kron(F, F)
    N = 4
    B = np.zeros((N, N), dtype=int)
    for i in range(N):
        B[i, bit_reversed_index(i, 2)] = 1
    x_ref = np.mod(u @ B @ G, 2)
    assert np.array_equal(x, x_ref), f"编码器错误: {x}, 期望 {x_ref}"
    print("Encoder test passed.")
