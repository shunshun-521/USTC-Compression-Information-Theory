"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)])


def polar_encode(u):
    """
    极化码编码（Kronecker 生成矩阵 F^{\\otimes n}，与 SC 译码器索引一致）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    v = np.asarray(u, dtype=int).copy()
    N = len(v)
    n = int(np.log2(N))
    for layer in range(1, n + 1):
        block = 1 << layer
        half = block // 2
        for kk in range(N // block):
            base = kk * block
            left = slice(base, base + half)
            right = slice(base + half, base + block)
            v[left] = v[left] ^ v[right]
    return v


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("N=4 encode:", x)
