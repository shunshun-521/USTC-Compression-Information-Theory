"""
极化码编码器
编码：x = B_N * F^(otimes n) * u，蝶形变换后做比特倒序置换
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)])


def _butterfly_transform(u):
    """Arikan 蝶形极化变换（O(N log N)）"""
    u = np.array(u, dtype=np.int8, copy=True)
    n = len(u)
    step = n
    while step > 1:
        half = step // 2
        for base in range(0, n, step):
            for k in range(half):
                idx = base + k
                u[idx] ^= u[idx + half]
        step = half
    return u


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    x = _butterfly_transform(u)
    rev = bit_reversal_permutation(len(x))
    return x[rev]


def polar_encode_core(u):
    """仅蝶形变换（不含比特倒序），供 BP 早停重编码使用"""
    return _butterfly_transform(u)


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u, "-> x =", x)
