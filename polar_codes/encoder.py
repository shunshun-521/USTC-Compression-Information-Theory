"""
极化码编码器
编码：x = u * G_N，块蝶形结构 O(N log N)
与 Arikan 生成矩阵 F^{\\otimes n}（F=[[1,1],[0,1]]）及 u 行向量约定一致
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        r = 0
        v = i
        for _ in range(n):
            r = (r << 1) | (v & 1)
            v >>= 1
        rev[i] = r
    return rev


def polar_encode(u):
    """
    极化码编码（块蝶形，与 3GPP / 常用参考实现一致）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位，冻结位应为 0）

    返回：
        x: 长度为 N 的码字（原地蝶形后的 u）
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    nblk = N
    while nblk > 1:
        n_split = nblk // 2
        for p in range(0, N, nblk):
            for k in range(n_split):
                u[p + k] ^= u[p + k + n_split]
        nblk = n_split
    return u.astype(int)


def polar_generator_matrix(N):
    """生成矩阵 G_N = F^{\\otimes n}，F=[[1,1],[0,1]]"""
    F = np.array([[1, 1], [0, 1]], dtype=int)
    G = F.copy()
    for _ in range(int(np.log2(N)) - 1):
        G = np.kron(G, F)
    return G


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix(4)
    print("encode:", x)
    print("u@G:", (u @ G) % 2)
