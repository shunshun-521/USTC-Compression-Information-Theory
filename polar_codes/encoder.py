"""
极化码编码器
编码采用分阶段 XOR-gather 结构（与 5G/Sionna 一致）。
"""
import numpy as np

_GEN_CACHE = {}


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组。"""
    n = int(np.log2(N))
    indices = np.arange(N, dtype=np.int64)
    rev = np.zeros(N, dtype=np.int64)
    for bit in range(n):
        rev = (rev << 1) | ((indices >> bit) & 1)
    return rev


def _gen_encode_indices(N):
    """预计算编码阶段的 gather 索引。"""
    nb_stages = int(np.log2(N))
    ind_gather = np.ones((nb_stages, N + 1), dtype=np.int32) * N
    for s in range(nb_stages):
        ind_range = np.arange(N // 2)
        ind_dest = ind_range * 2 - np.mod(ind_range, 2 ** s)
        ind_origin = ind_dest + 2 ** s
        ind_gather[s, ind_dest] = ind_origin
    return ind_gather


def polar_encode(u, info_indices=None):
    """
    极化码编码。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位），或
        info_indices: 若提供，则 u 仅含信息位（用于内部调用）
    """
    if info_indices is not None:
        payload = np.asarray(u, dtype=np.int8)
        N = 2 ** int(np.ceil(np.log2(max(info_indices.max() + 1, len(payload)))))
        full = np.zeros(N, dtype=np.int8)
        full[info_indices] = payload
        u = full

    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    n = int(np.log2(N))

    if N not in _GEN_CACHE:
        _GEN_CACHE[N] = _gen_encode_indices(N)

    x = np.zeros(N + 1, dtype=np.int8)
    x[:N] = u
    ind_gather = _GEN_CACHE[N]

    for s in range(n):
        ind_helper = ind_gather[s, :]
        x_add = x[ind_helper]
        x = np.bitwise_xor(x, x_add)

    return x[:N].astype(int) % 2


def polar_generator_matrix(N):
    """构造生成矩阵（用于校验）。"""
    n = int(np.log2(N))
    G = np.eye(N, dtype=np.int8)
    for i in range(N):
        e = np.zeros(N, dtype=np.int8)
        e[i] = 1
        G[i] = polar_encode(e)
    return G


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1, 0, 0, 0, 0])
    x = polar_encode(u)
    print(f"u={u} -> x={x}")
    print("编码器运行完成")
