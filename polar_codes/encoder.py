"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(format(i, f"0{n}b")[::-1], 2)
    return rev


def polar_encode(u):
    """
    极化码编码。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字

    实现：分层蝶形结构，等价于 x = u * G_N（G_N = F^{⊗n}）
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    m = 1
    for _ in range(int(np.log2(N))):
        for i in range(0, N, 2 * m):
            x_block = u[i : i + m].copy()
            y_block = u[i + m : i + 2 * m].copy()
            for j in range(m):
                u[i + j] = x_block[j] ^ y_block[j]
            u[i + m : i + 2 * m] = y_block
        m *= 2
    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u)
    print("x =", x)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
    print("编码器测试通过")
