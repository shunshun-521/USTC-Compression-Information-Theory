"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev_idx = np.zeros(N, dtype=int)
    for i in range(N):
        r = 0
        for b in range(n):
            if (i >> b) & 1:
                r |= 1 << (n - 1 - b)
        rev_idx[i] = r
    return rev_idx


def _gen_gather_indices(N: int) -> np.ndarray:
    """阶段式 XOR 编码索引（与标准极化码因子图一致）"""
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
        u: 长度为 N 的源序列（信息位 + 冻结位），或长度为 K 的信息比特（需提供 info_indices）
        info_indices: 信息位索引；若提供则 u 仅含 K 个信息比特

    返回：
        x: 长度为 N 的码字
    """
    if info_indices is not None:
        info_indices = np.asarray(info_indices, dtype=int)
        N = 1 << int(np.ceil(np.log2(len(info_indices) + (u.size if u.size else 1))))
        # N 由 info_indices 最大值 +1 推断
        N = int(2 ** np.ceil(np.log2(info_indices.max() + 1)))
        if N < info_indices.max() + 1:
            N = info_indices.max() + 1
            N = 1 << int(np.ceil(np.log2(N)))
        full = np.zeros(N, dtype=int)
        full[info_indices] = np.asarray(u, dtype=int)
        u = full

    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError("N must be a power of 2")

    x = np.zeros(N + 1, dtype=np.uint8)
    x[:N] = u.astype(np.uint8)
    ind_gather = _gen_gather_indices(N)
    for s in range(n):
        ind_helper = ind_gather[s, :]
        x_add = x[ind_helper]
        x = np.bitwise_xor(x, x_add)
    return x[:N].astype(int)


def polar_generator_matrix(N):
    """生成 G_N（用于校验）"""
    n = int(np.log2(N))
    u = np.eye(N, dtype=int)
    cols = [polar_encode(u[i]) for i in range(N)]
    return np.array(cols, dtype=int)


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    info = np.array([0, 1, 2, 3])
    x = polar_encode(u)
    print("u=", u, "x=", x)
