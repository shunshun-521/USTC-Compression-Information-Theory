"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f'0{n}b')[::-1], 2) for i in range(N)])


def polar_encode(u):
    """
    极化码编码。

    采用蝶形 XOR 结构（与 G_N = F^{⊗n} 等价），
    比特倒序置换由译码器端的索引顺序统一处理。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    block = N
    while block > 1:
        half = block // 2
        for base in range(0, N, block):
            for k in range(half):
                idx = base + k
                u[idx] ^= u[idx + half]
        block = half
    return u


def polar_encode_with_bitrev(u):
    """蝶形编码后再做比特倒序置换（与规范描述一致）。"""
    x = polar_encode(u)
    return x[bit_reversal_permutation(len(x))]


if __name__ == "__main__":
    u = np.array([0, 1, 0, 1])
    x = polar_encode(u)
    print("u =", u, "-> x =", x)
    assert np.array_equal(x, [0, 0, 1, 1]), f"编码器错误: {x}"
    print("Encoder test passed.")
