"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def polar_encode_pipelined(u):
    """Arikan 流水线极化编码（与 5G/srsRAN 一致，偶奇分解）。 """
    output = np.asarray(u, dtype=np.int8).copy()
    n = int(np.log2(len(output)))
    N = len(output)
    even = np.arange(0, N, 2)
    odd = np.arange(1, N, 2)
    tmp = np.zeros(N, dtype=np.int8)
    for _ in range(n):
        half = N // 2
        for j in range(half):
            tmp[j] = output[even[j]]
            tmp[j + half] = output[odd[j]]
        for j in range(half):
            output[odd[j]] = tmp[odd[j]]
            output[even[j]] = tmp[even[j]] ^ tmp[odd[j]]
    return output.astype(int)


def polar_encode(u):
    """
    极化码编码（蝶形 + 比特倒序置换，课程实验约定）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            for j in range(i, i + step):
                u[j] ^= u[j + step]
        step *= 2
    br = bit_reversal_permutation(N)
    return u[br].astype(int)


def polar_generator_matrix(N):
    """生成矩阵 G_N = B_N F^{\\otimes n}（与 polar_encode 一致）。"""
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    br = bit_reversal_permutation(N)
    B = np.eye(N, dtype=int)[br]
    return (B @ G) % 2
