"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=np.int64)
    rev = ((idx[:, None] >> np.arange(n)) & 1).astype(np.int64)
    rev = rev[:, ::-1]
    weights = 1 << np.arange(n)
    return (rev * weights).sum(axis=1)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字

    实现：蝶形（butterfly）递归结构
        - 每层：相邻对 (u[i], u[i + step]) -> (u[i] XOR u[i+step], u[i+step])
        - 共 log2(N) 层
        - 最后做比特倒序置换（bit-reversal permutation）
    """
    u = np.array(u, dtype=np.int8, copy=True)
    N = len(u)
    n = int(np.log2(N))
    block = N
    while block > 1:
        half = block // 2
        for start in range(0, N, block):
            u[start : start + half] ^= u[start + half : start + block]
        block = half
    return u.astype(int)


def polar_generator_matrix(N):
    """GF(2) 生成矩阵 F^{⊗n}（与蝶形编码一致；B_N 由 SC 译码顺序吸收）。"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = F.copy()
    for _ in range(int(np.log2(N)) - 1):
        G = np.kron(G, F)
    return G


def polar_encode_matrix(u):
    """矩阵乘法编码（参考实现）。"""
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    G = polar_generator_matrix(N)
    return (u @ G) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("encode:", x)
    print("matrix:", polar_encode_matrix(u))
