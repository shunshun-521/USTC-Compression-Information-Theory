"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.arange(N)
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in indices], dtype=int)


def polar_encode(u):
    """
    极化码编码（蝶形结构，与标准 F^{\\otimes n} 生成矩阵一致）。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = N
    while n > 1:
        n_split = n // 2
        for p in range(0, N, n):
            for k in range(n_split):
                l = p + k
                u[l] ^= u[l + n_split]
        n = n_split
    return u


def verify_encoder():
    """编码器单元测试（u * G_N，G_N = F^{\\otimes n}）"""
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"


if __name__ == "__main__":
    verify_encoder()
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print(f"u={u}, x={x}")
