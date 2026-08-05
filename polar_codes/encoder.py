"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
G_N = F^{\\otimes n}（自然索引顺序，与 SC 译码器一致）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码。

    蝶形结构（与 Kronecker 生成矩阵 F^{\\otimes n} 一致）：
        每层将下半段 XOR 到上半段：x[i:i+step] ^= x[i+step:i+2*step]
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            u[i:i + step] ^= u[i + step:i + 2 * step]
        step *= 2
    return u


def polar_generator_matrix(N):
    """返回 GF(2) 生成矩阵 G_N = F^{\\otimes n}。"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    n = int(np.log2(N))
    for _ in range(n - 1):
        top = np.hstack([G, np.zeros((G.shape[0], G.shape[1]), dtype=int)])
        bottom = np.hstack([G, G])
        G = np.vstack([top, bottom])
    return G


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix(4)
    x_mat = (u @ G) % 2
    print("u:", u)
    print("polar_encode:", x)
    print("u @ G:", x_mat)
    print("match:", np.array_equal(x, x_mat))
