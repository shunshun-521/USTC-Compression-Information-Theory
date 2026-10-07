"""
极化码编码器
编码：蝶形（butterfly）结构 O(N log N)，与标准 Arikan 非系统化编码一致
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
    极化码编码（蝶形 XOR，结果存于原向量索引顺序）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位，冻结位为 0）

    返回：
        x: 长度为 N 的码字（与 u 同索引域，编码后即为发送码字）
    """
    u = np.array(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    assert 2 ** n == N

    m = N
    for _ in range(n):
        if m == 1:
            break
        half = m // 2
        for base in range(0, N, m):
            for k in range(half):
                u[base + k] ^= u[base + k + half]
        m = half

    return u.astype(int)


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("encode test:", x)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
