"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np

_G_CACHE = {}


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(format(i, f"0{n}b")[::-1], 2)
    return rev


def build_generator_matrix(N):
    """构造 G_N = B_N * F^{⊗ n}（模 2）。"""
    if N in _G_CACHE:
        return _G_CACHE[N]

    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError("N must be a power of 2")

    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = np.array([[1]], dtype=np.int8)
    for _ in range(n):
        G = np.kron(G, F)

    rev = bit_reversal_permutation(N)
    B = np.zeros((N, N), dtype=np.int8)
    for i, j in enumerate(rev):
        B[i, j] = 1

    G = (B @ G) % 2
    _G_CACHE[N] = G
    return G


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError("N must be a power of 2")

    # 蝶形递归：与 G_N 矩阵乘法等价
    v = u.copy()
    step = N // 2
    while step >= 1:
        for i in range(0, N, 2 * step):
            left = v[i:i + step]
            right = v[i + step:i + 2 * step]
            v[i:i + step] = (left ^ right) & 1
        step //= 2

    rev = bit_reversal_permutation(N)
    x = v[rev]

    # 与生成矩阵结果交叉验证（仅小码长）
    if N <= 1024:
        G = build_generator_matrix(N)
        x_ref = (u.astype(np.int64) @ G) % 2
        if not np.array_equal(x, x_ref):
            raise ValueError("蝶形编码与生成矩阵不一致")

    return x


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    print("u =", u)
    print("x =", x)
    print("G @ u =", (u @ G) % 2)
    assert np.array_equal(x, (u @ G) % 2), f"编码器错误: {x}"
    print("编码器校验通过")
