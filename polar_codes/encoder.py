"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def _bit_reversed(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= (1 << (n - 1 - i))
    return result


def polar_encode(u):
    """
    极化码编码（蝶形 XOR，与标准 G_N = F^otimes n 一致）。
    """
    u = np.array(u, dtype=int).copy()
    n_len = len(u)
    block = n_len
    for _ in range(n_len):
        if block == 1:
            break
        half = block // 2
        for p in range(0, n_len, block):
            for k in range(half):
                u[p + k] ^= u[p + k + half]
        block = half
    return u


def polar_encode_no_br(u):
    """与 polar_encode 相同（保留接口兼容性）。"""
    return polar_encode(u)


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("polar_encode:", x)
