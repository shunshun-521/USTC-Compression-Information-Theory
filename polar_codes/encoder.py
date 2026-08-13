"""
极化码编码器
编码：利用蝶形（con）结构实现 O(N log N) 复杂度
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

    逐层执行 u[i:i+m] XOR u[i+m:i+2m] 并保留右半部分，
    等价于标准极化变换 F^{⊗n}。
    """
    u = np.array(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N)) + 1
    m = 1
    for _ in range(n - 1):
        for i in range(0, N, 2 * m):
            left = u[i:i + m].copy()
            right = u[i + m:i + 2 * m].copy()
            u[i:i + m] = left ^ right
            u[i + m:i + 2 * m] = right
        m *= 2
    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u, "-> x =", x)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
    print("Encoder test passed.")
