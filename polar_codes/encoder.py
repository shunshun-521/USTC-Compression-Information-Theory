"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组：out[i] = in[indices[i]]"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        r = 0
        for b in range(n):
            if (i >> b) & 1:
                r |= 1 << (n - 1 - b)
        rev[i] = r
    return rev


def polar_encode(u, apply_bit_reversal=False):
    """
    Arikan 极化变换：x = u * G_N（GF(2)）。
    内层蝶形 u[j] ^= u[j+s]，s = 1,2,...,N/2。
    可选输出比特倒序置换。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    for d in range(n):
        s = 1 << d
        for i in range(0, N, 2 * s):
            for j in range(i, i + s):
                u[j] ^= u[j + s]

    if apply_bit_reversal:
        u = u[bit_reversal_permutation(N)]
    return u.astype(int)


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    print("encode:", polar_encode(u))
    print("encode+BR:", polar_encode(u, apply_bit_reversal=True))
