"""
极化码编码器
编码：x = G_N * u，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.arange(N)
    rev = ((indices & 1) << (n - 1))
    for i in range(1, n):
        rev |= ((indices >> i) & 1) << (n - 1 - i)
    return rev


def polar_encode(u):
    """
    极化码编码（Kronecker 生成矩阵 G_N = F^{\\otimes n}）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.array(u, dtype=np.int8, copy=True)
    N = len(u)
    n = int(np.log2(N))
    for stage in range(n):
        m = 1 << stage
        stride = m << 1
        for i in range(0, N, stride):
            for j in range(m):
                u[i + j] ^= u[i + j + m]
    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u, "-> x =", x)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
    print("Encoder test passed.")
