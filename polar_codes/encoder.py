"""
极化码编码器
编码：x = u * F_N，F_N = F^{\\otimes n}，Arikan 蝶形结构 O(N log N)
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（Arikan 非递归蝶形，与 PSC 译码配套）。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    block = N
    while block > 1:
        half = block // 2
        for base in range(0, N, block):
            for k in range(half):
                u[base + k] ^= u[base + k + half]
        block = half
    return u


def polar_generator_matrix(N):
    """生成 F^{\\otimes n}（与 polar_encode 一致）"""
    F = np.array([[1, 1], [0, 1]], dtype=int)
    G = F.copy()
    nn = int(np.log2(N))
    for _ in range(nn - 1):
        G = np.kron(G, F)
    return G % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix(4)
    x_mat = (u @ G) % 2
    print("polar_encode([1,0,1,1]) =", x)
    print("matrix encode:", x_mat)
    print("match:", np.array_equal(x, x_mat))
