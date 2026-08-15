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
        r = 0
        for j in range(n):
            if (i >> j) & 1:
                r |= 1 << (n - 1 - j)
        rev[i] = r
    return rev


def polar_encode(u):
    """
    极化码编码。

    采用分块蝶形结构（与标准极化码生成矩阵一致），
    复杂度 O(N log N)。编码后码字即为信道传输序列。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    block = N

    while block > 1:
        half = block // 2
        for start in range(0, N, block):
            for k in range(half):
                idx = start + k
                u[idx] ^= u[idx + half]
        block = half

    return u.astype(int)


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u)
    print("x =", x)
    # 与生成矩阵 G_N 手算结果一致
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
    print("Encoder test passed!")
