"""
极化码编码器
编码：x = u * F_N（Kronecker 积，无比特倒序，与 SC/SCL 译码器一致）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(format(i, f"0{n}b")[::-1], 2)
    return rev


def polar_encode(u):
    """
    极化码编码（蝶形结构，O(N log N)）。
    采用 F^{\\otimes n} 变换（不含额外比特倒序），与译码器匹配。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            u[i : i + step] ^= u[i + step : i + 2 * step]
        step <<= 1
    return u


def polar_decode_u(x_hat):
    """从硬判决码字恢复源序列（逆蝶形）"""
    x_hat = np.asarray(x_hat, dtype=np.int8).copy()
    N = len(x_hat)
    u = x_hat.copy()
    step = N // 2
    while step >= 1:
        for i in range(0, N, 2 * step):
            u[i : i + step] ^= u[i + step : i + 2 * step]
        step //= 2
    return u


def build_generator_matrix(N):
    """构建生成矩阵（含比特倒序 B_N，用于验证）"""
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    rev = bit_reversal_permutation(N)
    return G[rev, :]


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u:", u, "-> x:", x)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
    assert np.array_equal(polar_decode_u(x), u), "逆编码失败"
