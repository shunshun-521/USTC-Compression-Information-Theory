"""
极化码编码器
编码：x = u * F_N（蝶形），与译码器使用相同的 Arikan 核约定
"""
import numpy as np


def _bitrev_indices(N):
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=np.int64)


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    return _bitrev_indices(N)


def polar_encode(u):
    """
    极化码编码：对 u 执行蝶形变换得到码字 x。
    （与置换 SC 译码器配套；内部包含与 B_N 等价的比特倒序重排。）
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")

    n = int(np.log2(N))
    block = N
    while block > 1:
        half = block // 2
        for start in range(0, N, block):
            for k in range(half):
                i = start + k
                u[i] ^= u[i + half]
        block = half

    br = _bitrev_indices(N)
    return u[br]


def polar_encode_butterfly_only(u):
    """仅蝶形，不做比特倒序（调试用）"""
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    block = N
    while block > 1:
        half = block // 2
        for start in range(0, N, block):
            for k in range(half):
                i = start + k
                u[i] ^= u[i + half]
        block = half
    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    print("u=", u, "x=", polar_encode(u))
