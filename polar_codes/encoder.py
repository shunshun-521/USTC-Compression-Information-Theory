"""
极化码编码器
编码：x = u * G_N，G_N = B_N * F^{\otimes n}
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
    蝶形结构：从大块到小块依次 XOR（与 Arikan 生成矩阵一致）
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    block = N
    while block > 1:
        half = block // 2
        for p in range(0, N, block):
            for k in range(half):
                l = p + k
                u[l] = u[l] ^ u[l + half]
        block = half

    rev = bit_reversal_permutation(N)
    x = u[rev]
    return x.astype(int)


def build_generator_matrix(N):
    """构建 G_N = B_N * F^{\otimes n}"""
    F = np.array([[1, 1], [0, 1]], dtype=int)
    G = F.copy()
    n = int(np.log2(N))
    for _ in range(n - 1):
        G = np.kron(F, G)
    rev = bit_reversal_permutation(N)
    Bn = np.zeros((N, N), dtype=int)
    for i in range(N):
        Bn[rev[i], i] = 1
    return (Bn @ G) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    x_ref = (u @ G) % 2
    print("u =", u, "-> x =", x, "ref =", x_ref)
    assert np.array_equal(x, x_ref), f"编码器错误: {x} != {x_ref}"
