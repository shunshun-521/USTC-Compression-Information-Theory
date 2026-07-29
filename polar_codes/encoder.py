"""
极化码编码器
编码：蝶形结构 O(N log N)，输出含比特倒序置换
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组。"""
    n = int(np.log2(N))
    return np.array(
        [_bit_reversed(i, n) for i in range(N)], dtype=int
    )


def _bit_reversed(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= (1 << (n - 1 - i))
    return result


def _butterfly_encode(u):
    """蝶形编码（不含比特倒序）。"""
    N = len(u)
    n = int(np.log2(N))
    x = u.copy()
    block = N
    while block > 1:
        half = block // 2
        for p in range(0, N, block):
            for k in range(half):
                l = p + k
                x[l] ^= x[l + half]
        block = half
    return x


def polar_encode(u):
    """
    极化码编码：蝶形 + 比特倒序置换。
    """
    u = np.asarray(u, dtype=np.int8)
    x = _butterfly_encode(u)
    rev = bit_reversal_permutation(len(u))
    return x[rev]


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    # 编码为自逆变换；验证 roundtrip
    x2 = polar_encode(x)
    assert np.array_equal(x2, u), f"自逆检验失败: {x2}"
    print("encoder roundtrip OK for u=[1,0,1,1]")
