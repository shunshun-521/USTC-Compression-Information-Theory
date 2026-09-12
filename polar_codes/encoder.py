r"""
极化码编码器
编码：x = u * F^{\otimes n}，蝶形 XOR 结构 O(N log N)
"""
import numpy as np

_G_CACHE = {}


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def _build_generator_matrix(N):
    """构造 G_N = F^{\otimes n}，行向量编码 x = u @ G_N"""
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    Fn = F.copy()
    for _ in range(n - 1):
        Fn = np.kron(Fn, F)
    return Fn % 2


def get_generator_matrix(N):
    if N not in _G_CACHE:
        _G_CACHE[N] = _build_generator_matrix(N)
    return _G_CACHE[N]


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
    G = get_generator_matrix(N)
    return (u @ G) % 2


def polar_encode_butterfly(u):
    """O(N log N) 蝶形编码，与矩阵乘法结果一致"""
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    x = u.copy()
    for stage in range(n):
        step = 1 << stage
        for i in range(0, N, step << 1):
            for j in range(step):
                x[i + j] = (x[i + j] + x[i + j + step]) % 2
    return x


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u)
    print("x =", x)
    x_bf = polar_encode_butterfly(u)
    print("butterfly x =", x_bf)
    assert np.array_equal(x, x_bf), "butterfly mismatch"
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
