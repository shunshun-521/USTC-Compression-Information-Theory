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
        rev[i] = int(format(i, f'0{n}b')[::-1], 2)
    return rev


def _bit_reversed(x, n):
    """对标量索引做比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def polar_encode(u):
    """
    极化码编码（蝶形结构，O(N log N)）。

    采用与 SC 译码器一致的极化变换：从大块到小块依次执行
    u[i] ^= u[i + block_size]，等价于 u 乘以 F 的 n 次 Kronecker 积。
    信道端码字顺序与译码器 L[:, 0] 的 likelihood 顺序一致。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    block = N
    while block > 1:
        half = block // 2
        for start in range(0, N, block):
            for k in range(half):
                idx = start + k
                u[idx] ^= u[idx + half]
        block = half
    return u


def polar_encode_with_bitrev(u):
    """带比特倒序置换的编码（与部分教材定义一致）"""
    x = polar_encode(u)
    rev = bit_reversal_permutation(len(x))
    return x[rev]


if __name__ == "__main__":
    u = np.array([0, 1, 0, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [0, 0, 1, 1]), f"编码器错误: {x}"
    print("Encoder test passed:", x)
