"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(f"{i:0{n}b}"[::-1], 2)
    return rev


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    蝶形结构与参考 polar-codes 库一致：从块长 N 向下递归合并。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n_block = N
    for _ in range(N):
        if n_block == 1:
            break
        n_split = n_block // 2
        for p in range(0, N, n_block):
            for k in range(n_split):
                l = p + k
                u[l] = u[l] ^ u[l + n_split]
        n_block = n_split

    br = bit_reversal_permutation(N)
    x = u[br]
    return x


def polar_generator_matrix(N):
    """生成极化码生成矩阵 G_N = B_N F^{otimes n}（用于验证）"""
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    F_n = F.copy()
    for _ in range(n - 1):
        F_n = np.kron(F_n, F)
    br = bit_reversal_permutation(N)
    B = np.eye(N, dtype=int)[br]
    return (B @ F_n) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u=", u, "x=", x)
    G = polar_generator_matrix(4)
    x_mat = (u @ G) % 2
    print("matrix encode:", x_mat)
