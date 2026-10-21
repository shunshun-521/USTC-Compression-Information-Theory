"""
极化码编码器
编码：u 经蝶形 XOR 得到码字 x（与因子图一致的常用实现，无输出比特倒序）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=np.int64)
    rev = np.zeros(N, dtype=np.int64)
    for i in idx:
        r = 0
        for b in range(n):
            if (i >> b) & 1:
                r |= 1 << (n - 1 - b)
        rev[i] = r
    return rev


def bit_reversed(i, n):
    """单索引比特倒序"""
    result = 0
    for b in range(n):
        if (i >> b) & 1:
            result |= 1 << (n - 1 - b)
    return result


def polar_encode(u):
    """
    极化码编码（蝶形 XOR，O(N log N)）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    block = N
    while block > 1:
        half = block // 2
        for start in range(0, N, block):
            for k in range(half):
                l = start + k
                u[l] ^= u[l + half]
        block = half
    return u


def polar_encode_with_bit_reversal(u):
    """可选：对码字做比特倒序置换（与部分教材定义一致）"""
    x = polar_encode(u)
    br = bit_reversal_permutation(len(x))
    return x[br]


def polar_encode_matrix(u):
    """参考：G_N = B_N F^{⊗ n} 矩阵乘法"""
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = np.array([[1]], dtype=np.int8)
    for _ in range(n):
        G = np.kron(G, F) % 2
    B = np.zeros((N, N), dtype=np.int8)
    br = bit_reversal_permutation(N)
    for i in range(N):
        B[i, br[i]] = 1
    GN = (B @ G) % 2
    return (u @ GN) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    print("butterfly:", polar_encode(u))
    print("matrix G@u:", polar_encode_matrix(u))
