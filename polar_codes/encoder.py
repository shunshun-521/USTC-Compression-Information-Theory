"""
极化码编码器
编码：Arikan 递归极化变换，x = u * G_N，复杂度 O(N log N)
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（Arikan 递归极化变换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.array(u, dtype=int)
    n = int(np.log2(len(u)))
    if 2**n != len(u):
        raise ValueError("Length of u must be a power of 2")

    if len(u) == 1:
        return u.copy()

    u_xor = (u[::2] ^ u[1::2]).astype(int)
    u_even = u[1::2].astype(int)
    return np.concatenate([polar_encode(u_xor), polar_encode(u_even)])


def polar_generator_matrix(N):
    """构造 G_N = B_N F^{\\otimes n}（用于验证）"""
    n = int(np.log2(N))
    f = np.array([[1, 0], [1, 1]], dtype=int)
    fn = f.copy()
    for _ in range(n - 1):
        fn = np.kron(fn, f)
    br = bit_reversal_permutation(N)
    bn = np.zeros((N, N), dtype=int)
    for i, j in enumerate(br):
        bn[i, j] = 1
    return (bn @ fn) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    g = polar_generator_matrix(4)
    print("polar_encode([1,0,1,1]) =", x)
    print("matrix encode =", (u @ g) % 2)
