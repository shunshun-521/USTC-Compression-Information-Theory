"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字（x = u @ F^{\otimes n} mod 2）

    实现：蝶形（butterfly）递归结构
        - 每层：u[i] ^= u[i + step]（step = 1, 2, 4, ...）
        - 共 log2(N) 层
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))

    for stage in range(n):
        step = 1 << stage
        for i in range(0, N, 2 * step):
            for j in range(i, i + step):
                u[j] ^= u[j + step]

    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print(f"u={u} -> x={x}")
    F4 = np.kron(np.array([[1, 0], [1, 1]]), np.array([[1, 0], [1, 1]]))
    expected = (u @ F4) % 2
    assert np.array_equal(x, expected), f"编码器错误: {x} != {expected}"
    print("Encoder test passed.")
