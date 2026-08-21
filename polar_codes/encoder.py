"""
极化码编码器
编码：x = u * G_N，G_N = F^{\\otimes n}（无输出比特倒序）
"""
import numpy as np

_G_CACHE = {}


def build_generator_matrix(N):
    """构造极化码生成矩阵 F^{\\otimes n}。"""
    if N in _G_CACHE:
        return _G_CACHE[N]

    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = F.copy()
    n = int(np.log2(N))
    for _ in range(n - 1):
        G = np.kron(G, F)

    _G_CACHE[N] = G
    return G


def polar_encode(u):
    """
    极化码编码：x = u @ G_N（蝶形结构 O(N log N)）。
    """
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    n = int(np.log2(N))
    out = u.copy()

    n_split = N
    for _ in range(n):
        n_split //= 2
        if n_split == 0:
            break
        for p in range(0, N, 2 * n_split):
            for k in range(n_split):
                l = p + k
                out[l] ^= out[l + n_split]

    return out.astype(int)


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    print(f"u={u}, butterfly x={x}, matrix x={(u @ G) % 2}")
