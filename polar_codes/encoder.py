"""
极化码编码器
编码：x = u * G_N，G_N = B_N F^{\\otimes n}，蝶形 O(N log N)
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组：out[i] = in[bit_reverse(i)]"""
    n = int(np.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def polar_encode_core(u):
    """蝶形编码（不含比特倒序），u 原地变换后即为 v = u F^{\\otimes n}。"""
    x = np.array(u, dtype=np.int8, copy=True)
    N = len(x)
    n = int(np.log2(N))
    for s in range(n):
        step = 1 << s
        for i in range(0, N, 2 * step):
            for j in range(step):
                x[i + j] ^= x[i + j + step]
    return x


def polar_encode(u):
    """
    极化码编码（含比特倒序置换 B_N）。
    """
    x = polar_encode_core(u)
    brp = bit_reversal_permutation(len(x))
    return x[brp]


def generator_matrix(N):
    """G_N = B_N F^{\\otimes n}（用于校验）。"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = np.array([[1]], dtype=np.int8)
    for _ in range(int(np.log2(N))):
        G = np.kron(G, F)
    G &= 1
    brp = bit_reversal_permutation(N)
    B = np.zeros((N, N), dtype=np.int8)
    for i, j in enumerate(brp):
        B[i, j] = 1
    return (B @ G) & 1


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = generator_matrix(4)
    x_ref = (u @ G) % 2
    print("u:", u, "x:", x, "matrix:", x_ref)
