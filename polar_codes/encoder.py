"""
极化码编码器（Sionna 兼容的 stage-gather XOR 结构）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组。"""
    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError("N must be a power of 2")
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def _gen_gather_indices(N: int) -> np.ndarray:
    """预计算各 stage 的 gather 索引（与 Sionna PolarEncoder 一致）。"""
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
        u: 长度为 N 的源向量（信息位 + 冻结位）
        info_indices: 若提供，则 u 仅为 K 个信息比特，将散射到对应位置

    返回：
        x: 长度为 N 的码字
    """
    if info_indices is not None:
        info_indices = np.asarray(info_indices, dtype=int)
        N = int(2 ** np.ceil(np.log2(info_indices.max() + 1)))
        c = np.zeros(N, dtype=np.uint8)
        c[info_indices] = np.asarray(u, dtype=np.uint8) & 1
    else:
        c = np.asarray(u, dtype=np.uint8).copy() & 1
        N = len(c)

    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError("N must be a power of 2")

    ind_gather = _gen_gather_indices(N)
    x = np.zeros(N + 1, dtype=np.uint8)
    x[:N] = c
    for s in range(n):
        ind_helper = ind_gather[s, :]
        x_add = x[ind_helper]
        x = np.bitwise_xor(x, x_add)
    return x[:N].astype(int)


def polar_encode_full(u_full):
    """对长度 N 的完整源向量编码（冻结位须已为 0）。"""
    return polar_encode(u_full, info_indices=None)


def polar_generator_matrix(N):
    """生成 G_N = B_N F^{\\otimes n}（GF(2)），用于校验。"""
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = np.array([[1]], dtype=int)
    for _ in range(n):
        G = np.kron(G, F)
    B = np.zeros((N, N), dtype=int)
    br = bit_reversal_permutation(N)
    for i, j in enumerate(br):
        B[i, j] = 1
    return (B @ G) % 2
