"""
极化码编码器
编码：x = u * G_N，G_N = F 的 n 次 Kronecker 积（Arikan 生成矩阵）
"""
import numpy as np

_G_CACHE = {}


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def build_generator_matrix(N):
    """构造 G_N = F 的 n 次 Kronecker 积（模 2）"""
    if N in _G_CACHE:
        return _G_CACHE[N]
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    n = int(np.log2(N))
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F) % 2
    _G_CACHE[N] = G.astype(np.int8)
    return _G_CACHE[N]


def polar_encode(u):
    """
    极化码编码：x = u * G_N（模 2）。
    与标准矩阵生成形式一致，便于 SC/BP 译码器对接。
    """
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    G = build_generator_matrix(N)
    return (u @ G) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u=", u, "x=", x)
    G = build_generator_matrix(4)
    assert np.array_equal(x, (u @ G) % 2)
    print("encoder OK")
