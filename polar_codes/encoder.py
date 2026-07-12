"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
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
        x: 长度为 N 的码字

    实现：蝶形（butterfly）递归结构
        - 每层：u[i+j] ^= u[i+j+step]
        - 共 log2(N) 层
        - 等价于 x = u @ F^{\\otimes n}（自然序）
    """
    u = np.array(u, dtype=np.int8, copy=True)
    n = int(np.log2(len(u)))
    for stage in range(n):
        step = 1 << stage
        for i in range(0, len(u), step * 2):
            for j in range(step):
                u[i + j] ^= u[i + j + step]
    return u.astype(int)


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u, "-> x =", x)
