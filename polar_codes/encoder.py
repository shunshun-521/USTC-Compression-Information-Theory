"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f'0{n}b')[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码。

    采用标准蝶形 XOR 结构（与 Arikan 生成矩阵 F^{\\otimes n} 一致）：
    自底向上逐层对相邻子块做 u_upper ^= u_lower。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    x = np.asarray(u, dtype=int).copy()
    n = int(np.log2(len(x)))
    for stage in range(1, n + 1):
        block = 1 << stage
        half = block >> 1
        for blk in range(len(x) // block):
            start = blk * block
            x[start:start + half] ^= x[start + half:start + block]
    return x


if __name__ == '__main__':
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print('u =', u)
    print('x =', x)
