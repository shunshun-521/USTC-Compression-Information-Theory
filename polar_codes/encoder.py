"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)])


def bit_reversed(x, n):
    """对标量索引 x 做 n 位比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def polar_encode(u):
    """
    极化码编码（蝶形结构，O(N log N)）。

    与 SC 译码器配套：编码不做输出比特倒序，译码在比特倒序索引顺序下进行。
    等价于 x = u * F^{\\otimes n} (mod 2)。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    block = N
    while block > 1:
        half = block // 2
        for base in range(0, N, block):
            for k in range(half):
                idx = base + k
                u[idx] ^= u[idx + half]
        block = half
    return u


def polar_encode_with_reversal(u):
    """带输出比特倒序置换的编码（供 BP 早停重编码等场景使用）。"""
    x = polar_encode(u)
    return x[bit_reversal_permutation(len(u))]


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u, "-> x =", x)
    # u * G_4 (G_4 = F^{\\otimes 2})
    F = np.array([[1, 0], [1, 1]])
    G4 = np.kron(F, F)
    assert np.array_equal(x, np.mod(u @ G4, 2)), f"编码器错误: {x}"
    print("编码器校验通过")
