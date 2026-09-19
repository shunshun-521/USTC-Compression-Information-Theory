"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array(
        [int(f"{i:0{n}b}"[::-1], 2) for i in range(N)],
        dtype=int,
    )


def polar_encode(u):
    """
    极化码编码：x = u @ (B_N F^{\\otimes n})。
    Arikan 蝶形后做比特倒序置换。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    n = int(np.log2(len(u)))
    step = 1
    for _ in range(n):
        for i in range(0, len(u), 2 * step):
            for j in range(step):
                u[i + j] ^= u[i + j + step]
        step <<= 1
    return u[bit_reversal_permutation(len(u))]


def build_generator_matrix(N):
    """构造 G_N = B_N F^{\\otimes n}，用于测试验证。"""
    n = int(np.log2(N))
    f = np.array([[1, 0], [1, 1]], dtype=np.int8)
    g = f.copy()
    for _ in range(n - 1):
        g = np.kron(g, f)
    br = bit_reversal_permutation(N)
    b = np.zeros((N, N), dtype=np.int8)
    for i in range(N):
        b[br[i], i] = 1
    return (b @ g) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    g = build_generator_matrix(4)
    x_mat = (u @ g) % 2
    print("u =", u)
    print("butterfly encode x =", x)
    print("matrix encode x =", x_mat)
