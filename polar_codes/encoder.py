"""
极化码编码器
编码：在信息位上放置比特后，按阶段 XOR（与 5G/Sionna 极化编码一致）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        r = 0
        v = i
        for _ in range(n):
            r = (r << 1) | (v & 1)
            v >>= 1
        rev[i] = r
    return rev


def _gen_encode_indices(n):
    """预计算各阶段 XOR 索引（与 Sionna PolarEncoder 一致）。"""
    ind_gather = np.ones([int(np.log2(n)), n + 1], dtype=np.int32) * n
    for s in range(int(np.log2(n))):
        ind_range = np.arange(n // 2)
        ind_dest = ind_range * 2 - np.mod(ind_range, 2**s)
        ind_origin = ind_dest + 2**s
        ind_gather[s, ind_dest] = ind_origin
    return ind_gather


def polar_encode(u, info_indices=None):
    """
    极化码编码。

    参数：
        u: 长度为 N 的源序列（冻结位为 0，信息位为待编码比特）
        info_indices: 信息位索引；若为 None 则假定 u 中非零位置即为信息位
                      （通常由调用方在 u 中仅填充信息位、其余为 0）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    n = int(np.log2(N))
    assert 2**n == N

    x = np.zeros(N, dtype=np.uint8)
    if info_indices is None:
        x[:] = u & 1
    else:
        x[info_indices] = u[info_indices] & 1

    ind_gather = _gen_encode_indices(N)
    for s in range(n):
        for j in range(N):
            origin = ind_gather[s, j]
            if origin < N:
                x[j] ^= x[origin]
    return x.astype(int)


def build_generator_matrix(N):
    """构建生成矩阵（用于验证）。"""
    n = int(np.log2(N))
    G = np.eye(N, dtype=int)
    ind_gather = _gen_encode_indices(N)
    for s in range(n):
        G_new = G.copy()
        for j in range(N):
            origin = ind_gather[s, j]
            if origin < N:
                G_new[j] ^= G[origin]
        G = G_new
    return G % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    info = np.array([0, 1, 2, 3])
    x = polar_encode(u, info)
    print("u=", u, "x=", x)
