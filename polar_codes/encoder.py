"""
极化码编码器
编码：x = u * G_N，G_N = F^{\\otimes n}，蝶形 O(N log N)
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组：out[i] = bit_reverse(i)"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=int)
    rev = np.zeros(N, dtype=int)
    for i in idx:
        r = 0
        v = i
        for _ in range(n):
            r = (r << 1) | (v & 1)
            v >>= 1
        rev[i] = r
    return rev


def polar_encode(u):
    """
    极化码编码。

    蝶形：u[l] ^= u[l + step]（左支 XOR），共 log2(N) 层。
    与 G_N = F^{\\otimes n} 的矩阵乘法 u @ G_N 等价。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    n = int(np.log2(len(u)))
    N = len(u)
    for stage in range(n):
        step = 1 << stage
        for left in range(0, N, 2 * step):
            right = left + step
            u[left:left + step] ^= u[right:right + step]
    return u.astype(int)


def build_generator_matrix(N):
    """构造 G_N = F^{\\otimes n}（用于测试）。"""
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    return G


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    x_ref = (u @ G) % 2
    print("u:", u)
    print("polar_encode:", x)
    print("matrix multiply:", x_ref)
    assert np.array_equal(x, x_ref), "encoder mismatch"
