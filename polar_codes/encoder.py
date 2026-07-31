"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    利用分块蝶形结构（等价于 F^{\\otimes n} 极化变换）：
      从块长 N 开始，逐层将块长减半，对每块执行 u[left] ^= u[right]
      最后对比特倒序置换。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    block_len = N
    n_layers = int(np.log2(N))

    for _ in range(n_layers):
        if block_len == 1:
            break
        half = block_len // 2
        for block_start in range(0, N, block_len):
            for k in range(half):
                idx = block_start + k
                u[idx] ^= u[idx + half]
        block_len = half

    br = bit_reversal_permutation(N)
    return u[br]


def generator_matrix(N):
    """返回与 polar_encode 一致的 G_N"""
    G = np.zeros((N, N), dtype=int)
    for j in range(N):
        ej = np.zeros(N, dtype=int)
        ej[j] = 1
        G[:, j] = polar_encode(ej)
    return G


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = generator_matrix(4)
    print("u:", u)
    print("polar_encode(u):", x)
    print("u @ G_N:", u @ G % 2)
    assert np.array_equal(x, u @ G % 2), "编码器与生成矩阵不一致"
