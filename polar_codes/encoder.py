"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def _bit_reversal_indices(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.arange(N, dtype=np.int64)
    rev = np.zeros(N, dtype=np.int64)
    for i in range(N):
        r = 0
        v = i
        for _ in range(n):
            r = (r << 1) | (v & 1)
            v >>= 1
        rev[i] = r
    return rev


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    return _bit_reversal_indices(N)


def _butterfly_encode(u):
    """蝶形编码（不含比特倒序）"""
    u = np.asarray(u, dtype=np.int64).copy()
    N = len(u)
    n = int(np.log2(N))
    block = N
    for _ in range(n):
        half = block // 2
        for p in range(0, N, block):
            for k in range(half):
                idx = p + k
                u[idx] ^= u[idx + half]
        block = half
    return u


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    蝶形：u[i] ^= u[i+step]，共 log2(N) 层，最后比特倒序置换。
    """
    encoded = _butterfly_encode(u)
    rev = _bit_reversal_indices(len(encoded))
    return encoded[rev]


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u:", u, "-> x:", x)
    # 标准蝶形+比特倒序编码结果
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"
