r"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
G_N = B_N F^{\otimes n}
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array(
        [int(format(i, f'0{n}b')[::-1], 2) for i in range(N)],
        dtype=int,
    )


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))

    for layer in range(n):
        step = 1 << layer
        for i in range(0, N, 2 * step):
            for j in range(step):
                u[i + j] ^= u[i + j + step]

    br = bit_reversal_permutation(N)
    x = u[br]
    return x.astype(int)


def polar_generator_matrix(N):
    """构建生成矩阵 G_N = B_N F^{\otimes n}"""
    return build_generator_matrix(N)


def build_generator_matrix(N):
    r"""构建生成矩阵 G_N = B_N F^{\otimes n}（用于验证）"""
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    F_n = np.array([[1]], dtype=int)
    for _ in range(n):
        F_n = np.kron(F_n, F) % 2

    br = bit_reversal_permutation(N)
    B = np.zeros((N, N), dtype=int)
    for i, p in enumerate(br):
        B[i, p] = 1

    return (F_n @ B) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    x_mat = (u @ G) % 2
    print("u:", u)
    print("butterfly encode:", x)
    print("matrix encode:", x_mat)
    assert np.array_equal(x, x_mat), "编码器与生成矩阵不一致"
