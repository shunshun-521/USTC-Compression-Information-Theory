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
    极化码编码（Arikan 蝶形，不做输出比特倒序）。
    与 SC 译码器（自然序 LLR）配套使用。
    """
    x = np.array(u, dtype=np.int8, copy=True)
    N = len(x)
    if N & (N - 1):
        raise ValueError("N must be power of 2")
    n = int(np.log2(N))
    for stage in range(n):
        step = 1 << stage
        for i in range(0, N, 2 * step):
            for j in range(step):
                x[i + j] ^= x[i + j + step]
    return x.astype(int)


def polar_encode_matrix(u):
    """基于生成矩阵 F^{\\otimes n} 的编码，用于校验。"""
    u = np.array(u, dtype=int)
    N = len(u)
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = np.array([[1]], dtype=int)
    for _ in range(n):
        G = np.kron(G, F)
    return (u @ G) % 2


def validate_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    x_ref = polar_encode_matrix(u)
    assert np.array_equal(x, x_ref), f"编码器与矩阵不一致: {x} vs {x_ref}"
    return x


if __name__ == "__main__":
    x = validate_encoder()
    print("N=4 encode test:", x)
