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
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    x = np.array(u, dtype=int).copy()
    N = len(x)
    n = int(np.log2(N))

    for layer in range(n - 1, -1, -1):
        step = 2 ** layer
        for i in range(0, N, 2 * step):
            for j in range(step):
                x[i + j] ^= x[i + j + step]

    perm = bit_reversal_permutation(N)
    return x[perm]


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u)
    print("x =", x)
    # 与生成矩阵 G_N = B_N F^{⊗n} 手算一致
    F = np.array([[1, 0], [1, 1]])
    G = np.kron(F, F)
    perm = bit_reversal_permutation(4)
    B = np.zeros((4, 4), int)
    for i, p in enumerate(perm):
        B[p, i] = 1
    x_expected = u @ (B @ G) % 2
    assert np.array_equal(x, x_expected), f"编码器错误: {x} != {x_expected}"
    print("Encoder test passed.")
