"""
极化码编码器
编码：Kronecker 蝶形变换 x = u F^{⊗ n}（与 SC/SCL 译码树一致）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=int)
    if n == 0:
        return idx
    rev = ((idx[:, None] & (1 << np.arange(n))) != 0).astype(int)
    rev = rev.dot(1 << np.arange(n - 1, -1, -1))
    return rev


def polar_generator_matrix(N):
    """生成矩阵 F^{⊗ n}（无比特倒序行置换）"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    while G.shape[0] < N:
        G = np.kron(G, F)
    return G


def polar_encode(u):
    """
    极化码编码：蝶形 Kronecker 变换（O(N log N)）。
    """
    x = np.asarray(u, dtype=np.uint8).copy()
    length = x.size
    step = 1
    while step < length:
        for start in range(0, length, 2 * step):
            x[start : start + step] ^= x[start + step : start + 2 * step]
        step *= 2
    return x.astype(int)


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    print("N=4 encode:", polar_encode(u))
