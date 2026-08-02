"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度

蝶形运算与比特倒序置换共同构成 G_N = B_N * F^(otimes n)。
实现上采用 O(N log N) 蝶形变换；比特倒序由译码器在译码顺序中处理，
与显式对码字做 B_N 等价。
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(f"{i:0{n}b}"[::-1], 2)
    return rev


def polar_encode(u):
    """
    极化码蝶形编码。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")

    block = N
    while block > 1:
        half = block // 2
        for base in range(0, N, block):
            for k in range(half):
                idx = base + k
                u[idx] ^= u[idx + half]
        block = half

    return u


def polar_encode_with_reversal(u):
    """显式比特倒序置换后的码字（用于矩阵校验）。"""
    x = polar_encode(u)
    return x[bit_reversal_permutation(len(x))]


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u)
    print("x =", x)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
    print("Encoder test passed.")
