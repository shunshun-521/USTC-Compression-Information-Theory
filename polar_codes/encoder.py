"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def prepare_channel_llr(llr_ch):
    """将自然顺序信道 LLR 变换为译码器使用的比特倒序顺序。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    return llr_ch[bit_reversal_permutation(N)]


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=np.int64)


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
    u = np.asarray(u, dtype=np.int8).copy()
    N = u.shape[0]
    if N & (N - 1):
        raise ValueError("N must be a power of 2")
    n = int(np.log2(N))
    for stage in range(n):
        step = 1 << stage
        tmp = u.copy()
        for i in range(0, N, 2 * step):
            for j in range(step):
                idx1 = i + j
                idx2 = i + j + step
                u[idx1] = tmp[idx1] ^ tmp[idx2]
                u[idx2] = tmp[idx2]
    br = bit_reversal_permutation(N)
    return u[br]


def polar_encode_matrix(u):
    """基于生成矩阵的编码（用于校验），与 polar_encode 结果一致。"""
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = np.array([[1]], dtype=np.int8)
    for _ in range(n):
        G = np.kron(G, F)
    br = bit_reversal_permutation(N)
    G = G[br, :]
    return (u @ G) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    xm = polar_encode_matrix(u)
    print("butterfly:", x)
    print("matrix:", xm)
    assert np.array_equal(x, xm)
