"""
极化码编码器
编码：x = u * G_N = u * F^{⊗n}，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字，x = u @ F^{⊗n} (mod 2)

    实现：蝶形 XOR，每层 u[j] ^= u[j + step]，共 log2(N) 层。
    与 G_N = B_N F^{⊗n} 配合比特倒序信道索引等价。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    for d in range(n):
        step = 1 << d
        for i in range(0, N, step << 1):
            for j in range(i, i + step):
                u[j] ^= u[j + step]
    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    # u @ F^{⊗2} = [1, 1, 0, 1]
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
    print("Encoder test passed:", x)
