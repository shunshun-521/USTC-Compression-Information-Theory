"""
极化码编码器
x = u * G_N，G_N = F^{\\otimes n}（GF(2)），蝶形 O(N log N)
"""
import numpy as np


def bit_reversal_permutation(N):
    """比特倒序置换索引（部分构造/分析工具使用）"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=np.int64)
    for i in range(N):
        r = 0
        for b in range(n):
            if (i >> b) & 1:
                r |= 1 << (n - 1 - b)
        rev[i] = r
    return rev


def polar_encode(u):
    """
    极化码编码：x = u * F^{\\otimes n}（无额外比特倒序，与标准生成矩阵一致）
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    for i in range(n):
        offset = 1 << i
        for j in range(offset):
            for k in range(1 << (n - i - 1)):
                idx = j + 2 * offset * k
                u[idx] ^= u[idx + offset]
    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    print("u=", u, "x=", polar_encode(u))
