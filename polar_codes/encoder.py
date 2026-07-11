"""
极化码编码器
编码：x = u * F^{\\otimes n}，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（Arikan 蝶形 XOR 结构，等价于 u @ F^{\\otimes n}）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError("N must be a power of 2")

    for stage in range(n):
        step = 1 << stage
        for i in range(0, N, 2 * step):
            for j in range(step):
                u[i + j] ^= u[i + j + step]
    return u


def build_generator_matrix(N):
    """构建 F^{\\otimes n}（用于校验）"""
    n = int(np.log2(N))

    def polar_f_power(k):
        if k == 0:
            return np.array([[1]], dtype=int)
        half = polar_f_power(k - 1)
        zeros = np.zeros_like(half)
        top = np.hstack([half, zeros])
        bottom = np.hstack([half, half])
        return np.vstack([top, bottom])

    return polar_f_power(n)


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    x_mat = (u @ G) % 2
    print("polar_encode:", x)
    print("matrix encode:", x_mat)
    print("match:", np.array_equal(x, x_mat))
