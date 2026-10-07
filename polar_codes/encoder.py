"""
极化码编码器
编码：x = u * F^{⊗n}（与译码器一致的非比特倒序约定）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组。"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=int)
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(format(i, f"0{n}b")[::-1], 2)
    return rev


def polar_encode(u):
    """
    极化码编码（Kronecker F 变换，GF(2)）。
    """
    x = np.asarray(u, dtype=np.int8).copy()
    N = len(x)
    step = 1
    while step < N:
        for start in range(0, N, 2 * step):
            x[start : start + step] ^= x[start + step : start + 2 * step]
        step *= 2
    return x


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("encode test:", x)
