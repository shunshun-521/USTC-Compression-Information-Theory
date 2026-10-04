"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组：x[i] = u[perm[i]]"""
    n = int(np.log2(N))
    if N != (1 << n):
        raise ValueError("N must be a power of two")
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码：x = u @ G_N，G_N = F^{\\otimes n}（行向量约定）。
    实现为分块蝶形 XOR，复杂度 O(N log N)。
    """
    v = np.asarray(u, dtype=np.int8).copy()
    N = len(v)
    n = int(np.log2(N))
    if N != (1 << n):
        raise ValueError("Length of u must be a power of two")

    block = N
    while block > 1:
        half = block // 2
        for base in range(0, N, block):
            for k in range(half):
                idx = base + k
                v[idx] ^= v[idx + half]
        block = half
    return v.astype(int)


def arikan_generator(N):
    """生成矩阵 G_N = F^{\\otimes n}（GF(2)），满足 polar_encode(u) = (u @ G_N) % 2。"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = F.copy()
    for _ in range(int(np.log2(N)) - 1):
        G = np.kron(G, F)
    return G


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = arikan_generator(4)
    x_ref = (u @ G) % 2
    print("u:", u, "x:", x, "u@G:", x_ref)
