"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f'0{n}b')[::-1], 2) for i in range(N)])


def polar_transform(u):
    """
    极化变换（蝶形结构，不含比特倒序）。
    等价于 u @ F^{\\otimes n}，F = [[1,0],[1,1]]
    """
    u = np.array(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    for layer in range(n):
        step = 1 << layer
        for i in range(0, N, 2 * step):
            for j in range(step):
                u[i + j] ^= u[i + j + step]
    return u


def polar_transform_inverse(x):
    """极化变换的逆（F 为自逆矩阵）"""
    return polar_transform(x)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    x = bit_reversal(u @ F^{\\otimes n})
    """
    u = np.array(u, dtype=int)
    N = len(u)
    encoded = polar_transform(u)
    br = bit_reversal_permutation(N)
    return encoded[br]


def build_generator_matrix(N):
    """构建极化码生成矩阵 G_N = B_N F^{\\otimes n}（用于验证）"""
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    A = F.copy()
    for _ in range(n - 1):
        A = np.kron(A, F)
    br = bit_reversal_permutation(N)
    return A[br, :] % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    x_mat = (u @ G) % 2
    print(f"u={u}")
    print(f"butterfly encode: {x}")
    print(f"matrix encode:    {x_mat}")
