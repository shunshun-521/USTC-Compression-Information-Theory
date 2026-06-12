r"""
极化码编码器
编码：x = u * G_N，G_N = F^{\otimes n}，蝶形结构 O(N log N)
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    r"""
    极化码编码：x = u * F^{\otimes n}（mod 2）。
    蝶形结构：逐层 u[j] ^= u[j + 2^s]（s = 0, ..., log2(N)-1）
    """
    u = np.array(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    for s in range(n):
        m = 1 << (s + 1)
        half = m >> 1
        for i in range(0, N, m):
            for j in range(i, i + half):
                u[j] = (u[j] + u[j + half]) % 2
    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u:", u, "-> x:", x)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
    print("encoder test passed")
