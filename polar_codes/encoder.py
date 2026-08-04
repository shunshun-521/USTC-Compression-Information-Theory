"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.arange(N, dtype=int)
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(format(i, f'0{n}b')[::-1], 2)
    return rev


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字

    实现：蝶形（butterfly）递归结构
        - 每层：相邻对 (u[i], u[i + step]) -> (u[i] XOR u[i+step], u[i+step])
        - 共 log2(N) 层
        - 最后做比特倒序置换（bit-reversal permutation）
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))

    step = 1
    for _ in range(n):
        for i in range(0, N, 2 * step):
            u[i:i + step] ^= u[i + step:i + 2 * step]
        step *= 2

    rev = bit_reversal_permutation(N)
    return u[rev]


def build_generator_matrix(N):
    """构建生成矩阵 G_N = B_N F^{⊗n}（用于验证）"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(int(np.log2(N)) - 1):
        G = np.kron(G, F)
    rev = bit_reversal_permutation(N)
    B = np.eye(N, dtype=int)[rev]
    return (B @ G) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print(f"u={u} -> x={x}")

    u2 = np.array([0, 0, 1, 1])
    x2 = polar_encode(u2)
    print(f"u={u2} -> x={x2}")
    assert np.array_equal(x2, [0, 0, 1, 1]), f"编码器错误: {x2}"

    G = build_generator_matrix(4)
    for test_u in [[1, 0, 1, 1], [0, 0, 1, 1]]:
        test_u = np.array(test_u)
        assert np.array_equal(polar_encode(test_u), test_u @ G % 2)
