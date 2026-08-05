"""
极化码编码器
编码：x = u * G_N，G_N = F^{\\otimes n}
"""
import numpy as np

from polar_common import generate_matrix


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组。"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(format(i, f"0{n}b")[::-1], 2)
    return rev


def polar_encode_butterfly(u):
    """蝶形结构编码（与矩阵编码等价的标准实现）。"""
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    step = 1
    for _ in range(n):
        for i in range(0, N, 2 * step):
            u[i] = u[i] ^ u[i + step]
        step *= 2
    return u


def polar_encode(u):
    """
    极化码编码。
    使用生成矩阵 G_N = F^{\\otimes n}，与蝶形结构等价。
    """
    u = np.asarray(u, dtype=int)
    N = len(u)
    n = int(np.log2(N))
    G = generate_matrix(n)
    return (u @ G) % 2


def _generator_matrix(N):
    """构造 G_N。"""
    n = int(np.log2(N))
    return generate_matrix(n)


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = _generator_matrix(4)
    x_ref = u @ G % 2
    print("u =", u, "-> x =", x, "(ref:", x_ref, ")")
    assert np.array_equal(x, x_ref), f"编码器错误: {x} != {x_ref}"
    print("Encoder test passed.")
