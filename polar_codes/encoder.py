"""
极化码编码器
编码：x = u * G_N，利用蝶形 XOR 结构实现 O(N log N) 复杂度
"""
import numpy as np

_GATHER_CACHE = {}


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def _gen_gather_indices(N):
    """Sionna 风格逐层 XOR 编码索引"""
    n = int(np.log2(N))
    ind_gather = np.ones((n, N + 1), dtype=np.int32) * N
    for s in range(n):
        ind_range = np.arange(N // 2)
        ind_dest = ind_range * 2 - np.mod(ind_range, 2**s)
        ind_origin = ind_dest + 2**s
        ind_gather[s, ind_dest] = ind_origin
    return ind_gather


def polar_encode(u):
    """
    极化码编码（Sionna / 3GPP 蝶形 XOR，与 SC 译码器一致）。

    参数：
        u: 长度为 N 的源序列（冻结位为 0）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.uint8).copy()
    N = len(u)
    n = int(np.log2(N))
    if N not in _GATHER_CACHE:
        _GATHER_CACHE[N] = _gen_gather_indices(N)
    ind_gather = _GATHER_CACHE[N]

    x = np.zeros(N + 1, dtype=np.uint8)
    x[:N] = u
    for s in range(n):
        x = np.bitwise_xor(x, x[ind_gather[s, :]])
    return x[:N].astype(int)


def get_generator_matrix(N):
    """显式生成矩阵 G_N（用于验证）；编码主路径使用 polar_encode"""
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    br = bit_reversal_permutation(N)
    B = np.zeros((N, N), dtype=np.int8)
    B[np.arange(N), br] = 1
    return (B @ G) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    print("u=", u, "x=", polar_encode(u))
