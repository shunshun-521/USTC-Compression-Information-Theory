"""
极化码编码器
采用分阶段 XOR 蝶形结构（与标准 PolarEncoder 一致），复杂度 O(N log N)
"""
import numpy as np


def _gen_gather_indices(n):
    """预计算各阶段的 gather 索引。"""
    nb = int(np.log2(n))
    ind = np.ones((nb, n + 1), dtype=np.int64) * n
    for s in range(nb):
        ind_range = np.arange(n // 2)
        ind_dest = ind_range * 2 - ind_range % (2 ** s)
        ind_origin = ind_dest + 2 ** s
        ind[s, ind_dest] = ind_origin
    return ind


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array(
        [int("".join(reversed(format(i, f"0{n}b"))), 2) for i in range(N)],
        dtype=np.int64,
    )


def polar_encode(u):
    """
    极化码编码。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位，冻结位为 0）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.uint8)
    n = len(u)
    if n & (n - 1):
        raise ValueError("N must be a power of 2")

    x = np.zeros(n + 1, dtype=np.uint8)
    x[:n] = u
    indices = _gen_gather_indices(n)
    for s in range(int(np.log2(n))):
        x = np.bitwise_xor(x, x[indices[s]])
    return x[:n].astype(np.int8)


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    print("u =", u, "-> x =", polar_encode(u))
