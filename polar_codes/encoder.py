"""
极化码编码器
编码：x = u * G_N，G_N = F^{\\otimes n}，蝶形结构 O(N log N)
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)])


def polar_encode(u):
    """
    极化码编码：x = u * G_N（F 的 n 次 Kronecker 积），蝶形 XOR 结构。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    step = 1
    n = int(np.log2(N))
    for _ in range(n):
        for i in range(0, N, 2 * step):
            for j in range(step):
                u[i + j] = (u[i + j] + u[i + j + step]) % 2
        step *= 2
    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
    print("Encoder test passed:", x)
