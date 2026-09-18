"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f'0{n}b')[::-1], 2) for i in range(N)])


def polar_encode(u):
    """
    极化码编码。

    蝶形结构：u[l] ^= u[l + n_split]（mod 2），共 log2(N) 层。
    与标准 Arikan 生成矩阵 F^⊗n 一致（F=[[1,1],[0,1]]）。
    """
    u = np.array(u, dtype=int).copy()
    N = len(u)
    n = N
    while n > 1:
        half = n // 2
        for p in range(0, N, n):
            for k in range(half):
                u[p + k] = (u[p + k] + u[p + k + half]) % 2
        n = half
    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u, "-> x =", x)
    # 手算验证：F^2 编码 u=[1,0,1,1] -> x=[1,1,0,1]
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
