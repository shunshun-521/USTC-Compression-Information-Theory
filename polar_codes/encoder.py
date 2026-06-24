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
    极化码编码（非系统化，自然序蝶形 XOR）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字

    实现：对 u 执行 log2(N) 层蝶形 XOR，等价于 x = u * F^{⊗n}。
    译码器在比特倒序索引顺序下完成 SC/SCL，整体等价于 G_N = B_N F^{⊗n}。
    """
    u = np.asarray(u, dtype=int)
    N = len(u)
    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError("N must be a power of 2")

    d = u.copy()
    for stage in range(n):
        step = 1 << stage
        for i in range(0, N, 2 * step):
            for j in range(i, i + step):
                d[j] ^= d[j + step]
    return d


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u)
    print("x =", x)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
    print("Encoder test passed.")
