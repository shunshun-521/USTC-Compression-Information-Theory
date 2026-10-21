"""
极化码编码器
编码：x = u * F_N（Kronecker 积），蝶形 O(N log N)
注：采用非比特倒序约定（x = u * F^{⊗n}），与 SC/SCL 因子图一致。
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        r = 0
        for b in range(n):
            if (i >> b) & 1:
                r |= 1 << (n - 1 - b)
        rev[i] = r
    return rev


def polar_encode(u):
    """
    极化码编码（蝶形结构）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.array(u, dtype=np.int8).copy()
    N = len(u)
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            u[i : i + step] ^= u[i + step : i + 2 * step]
        step *= 2
    return u.astype(int)


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("encode test:", x)
    # 手算 u * G_4, G_4 = [[1,1,1,1],[0,0,1,1],[0,1,0,1],[0,0,0,1]]
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"
