"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    return _br_indices(N)


def _br_indices(N):
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
    极化码编码（蝶形 XOR，与 SC 因子图一致的 Arikan 非系统编码）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            for j in range(i, i + step):
                u[j] ^= u[j + step]
        step *= 2
    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("encode test:", x)
    # 标准生成矩阵验证：x = [1,1,0,1]
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
