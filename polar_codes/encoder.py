"""
极化码编码器
编码：x = u * F_N（Kronecker 积），蝶形块 XOR，O(N log N)
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        r = 0
        v = i
        for _ in range(n):
            r = (r << 1) | (v & 1)
            v >>= 1
        rev[i] = r
    return rev


def polar_encode(u):
    """
    极化码编码（与 SC/SCL 译码器一致的 F^{⊗n} 变换，无额外比特倒序）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    x = np.asarray(u, dtype=np.int8).copy()
    N = len(x)
    step = 1
    while step < N:
        for start in range(0, N, 2 * step):
            x[start: start + step] ^= x[start + step: start + 2 * step]
        step <<= 1
    return x


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("encode:", x)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
