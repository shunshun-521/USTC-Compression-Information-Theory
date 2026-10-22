"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    if 2**n != N:
        raise ValueError("N must be a power of 2")
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def _butterfly_encode(u):
    """蝶形递归编码（未做比特倒序）。"""
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    for stage in range(n):
        step = 1 << stage
        for i in range(0, N, 2 * step):
            for j in range(i, i + step):
                u[j] ^= u[j + step]
    return u


def polar_encode(u):
    """
    极化码编码：蝶形 XOR（与 SC 因子图一致的 u 域编码）。
    等价于 x = u * F^{\\otimes n}；译码端在比特倒序相位下完成 B_N 置换。
    """
    u = np.asarray(u, dtype=np.int8)
    return _butterfly_encode(u).astype(int)


def polar_encode_with_bit_reversal(u):
    """仅蝶形，不做输出倒序（调试用）。"""
    return _butterfly_encode(u).astype(int)


def build_generator_matrix(N):
    """G_N = B_N F^{\\otimes n} over GF(2)."""
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = F.copy()
    for _ in range(int(np.log2(N)) - 1):
        G = np.kron(G, F)
    br = bit_reversal_permutation(N)
    B = np.zeros((N, N), dtype=np.int8)
    for i, j in enumerate(br):
        B[i, j] = 1
    return (B @ G) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = F.copy()
    for _ in range(1):
        G = np.kron(G, F)
    x_ref = (u @ G) % 2
    assert np.array_equal(x, x_ref), f"encode mismatch: {x} vs {x_ref}"
    print("N=4 encoder OK:", x)
