r"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
G_N = B_N * F^{\otimes n}，最后做比特倒序置换
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f'0{n}b')[::-1], 2) for i in range(N)])


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    等价于 x = u @ (B_N * F^(⊗ n)) mod 2
    """
    u = np.array(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))

    step = 1
    for _ in range(n):
        for i in range(0, N, 2 * step):
            for j in range(i, i + step):
                u[j] ^= u[j + step]
        step *= 2

    rev = bit_reversal_permutation(N)
    return u[rev]


def polar_encode_no_br(u):
    r"""无比特倒序的极化编码，x = u @ F^(\otimes n) mod 2（供内部校验）"""
    u = np.array(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    step = 1
    for _ in range(n):
        for i in range(0, N, 2 * step):
            for j in range(i, i + step):
                u[j] ^= u[j + step]
        step *= 2
    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    # 标准极化码 G_N = B_N * F^{\otimes 2}：x = [1,0,1,1]
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}"
    print("Encoder test PASSED:", u, "->", x)
