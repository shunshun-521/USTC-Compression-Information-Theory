"""
极化码编码器
编码：x = u * G_N，G_N = F^{\\otimes n}
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.arange(N, dtype=np.int32)
    rev = np.zeros(N, dtype=np.int32)
    for bit in range(n):
        rev |= ((indices >> bit) & 1) << (n - 1 - bit)
    return rev


def _generator_matrix(N):
    """Arikan 生成矩阵 G = F^{\\otimes n}"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int32)
    G = F.copy()
    n = int(np.log2(N))
    for _ in range(n - 1):
        G = np.kron(G, F)
    return G


def polar_encode(u):
    """
    极化码编码：x = u @ G_N (mod 2)
    等价于蝶形 XOR 的标准 Arikan 极化变换。
    """
    u = np.asarray(u, dtype=np.int32)
    N = len(u)
    n = int(np.log2(N))
    assert 2 ** n == N
    G = _generator_matrix(N)
    return (u @ G) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print(f"u={u} -> x={x}")
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
    print("Encoder test passed.")
