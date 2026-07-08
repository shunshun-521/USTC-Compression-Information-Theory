"""
极化码编码器
编码：x = u * G_N，G_N = B_N F^{\\otimes n}，蝶形 O(N log N) 实现
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def _butterfly_encode(u):
    """蝶形 XOR：每层 (u[i], u[i+step]) -> (u[i] XOR u[i+step], u[i+step])。"""
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    block = N
    while block > 1:
        half = block // 2
        for base in range(0, N, block):
            for k in range(half):
                idx = base + k
                u[idx] ^= u[idx + half]
        block = half
    return u


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字，x[i] = enc[bit_reverse(i)]
    """
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    enc = _butterfly_encode(u)
    br = bit_reversal_permutation(N)
    return enc[br]


def polar_generator_matrix(N):
    """返回生成矩阵 G_N = B_N F^{\\otimes n}，F=[[1,1],[0,1]]。"""
    n = int(np.log2(N))
    F = np.array([[1, 1], [0, 1]], dtype=np.int8)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    br = bit_reversal_permutation(N)
    B = np.zeros((N, N), dtype=np.int8)
    for j in range(N):
        B[br[j], j] = 1
    return (B @ G) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix(4)
    x_ref = (u @ G) % 2
    print("u:", u)
    print("polar_encode:", x)
    print("matrix multiply:", x_ref)
    assert np.array_equal(x, x_ref)
