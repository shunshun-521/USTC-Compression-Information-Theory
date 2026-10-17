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
        b = 0
        for j in range(n):
            if (i >> j) & 1:
                b |= 1 << (n - 1 - j)
        rev[i] = b
    return rev


def polar_encode(u):
    """
    极化码编码（蝶形结构，与 SC 译码器一致的 G_N = F^{⊗n} 约定）。

    注：比特倒序 B_N 已吸收到信息位索引选取（GA 构造）中，此处不再对码字重排。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    step = 1
    for _ in range(n):
        for i in range(0, N, 2 * step):
            u[i : i + step] ^= u[i + step : i + 2 * step]
        step *= 2
    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("encode test:", x)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
