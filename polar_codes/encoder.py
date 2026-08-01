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

    蝶形结构：每层对左半分区执行 u[l] ^= u[l + n_split]，
    译码器在比特倒序顺序下处理，与 bit-reversal 置换编码等价。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    assert 2 ** n == N

    block = N
    while block > 1:
        half = block // 2
        for start in range(0, N, block):
            for k in range(half):
                idx = start + k
                u[idx] ^= u[idx + half]
        block = half

    return u


def polar_encode_with_bitrev(u):
    """含比特倒序置换的编码（与生成矩阵 G_N = B_N F^{\\otimes n} 一致）"""
    v = polar_encode(u)
    return v[bit_reversal_permutation(len(u))]


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u, "-> x =", x)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
    print("Encoder test passed.")
