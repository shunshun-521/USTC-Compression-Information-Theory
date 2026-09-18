"""
极化码编码器
编码：x = u * F^{\\otimes n}，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def bit_reversed_index(i, n):
    """单索引比特倒序"""
    return int(format(i, f"0{n}b")[::-1], 2)


def polar_encode(u):
    """
    极化码编码（Arikan 蝶形 XOR，与 polarcodes 一致）。

    编码后码字 x 满足 x = u @ F^{\\otimes n} (mod 2)。
    比特倒序置换 B_N 由 SC 译码器的译码顺序吸收，不在编码端显式施加。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    n = int(np.log2(len(u)))
    block = len(u)
    for _ in range(n):
        half = block // 2
        for base in range(0, len(u), block):
            for k in range(half):
                idx = base + k
                u[idx] ^= u[idx + half]
        block = half
    return u.astype(int)


def source_from_codeword(x):
    """由码字 x 恢复源序列 u = x @ F^{\\otimes n} (mod 2)"""
    return polar_encode(x)


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u, "-> x =", x)
