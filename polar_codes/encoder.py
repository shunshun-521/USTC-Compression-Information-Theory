"""
极化码编码器
编码：x = u * G_N，G_N = F^{\\otimes n}（与 SC 因子图一致的蝶形结构，无输出比特倒序）
可选 bit-reversal 编码见 polar_encode_with_bit_reversal
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.arange(N, dtype=int)
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        r = 0
        v = i
        for _ in range(n):
            r = (r << 1) | (v & 1)
            v >>= 1
        rev[i] = r
    return rev


def _butterfly_encode(u):
    """蝶形 XOR 编码核，O(N log N)"""
    N = len(u)
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            u[i : i + step] ^= u[i + step : i + 2 * step]
        step <<= 1
    return u


def polar_encode(u):
    """
    极化码编码（标准蝶形，与译码器因子图一致）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.array(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    assert 2 ** n == N
    return _butterfly_encode(u).astype(int)


def polar_encode_with_bit_reversal(u):
    """蝶形编码后再做比特倒序置换"""
    x = polar_encode(u)
    return x[bit_reversal_permutation(len(x))]


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    print("polar_encode:", polar_encode(u))
