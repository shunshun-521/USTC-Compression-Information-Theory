"""
极化码编码器
编码：x = u * G_N，利用蝶形 XOR 结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    if n == 0:
        return np.array([0], dtype=int)
    idx = np.arange(N, dtype=np.int64)
    rev = np.zeros(N, dtype=np.int64)
    for bit in range(n):
        rev += ((idx >> bit) & 1) << (n - 1 - bit)
    return rev.astype(int)


def _gen_xor_indices(n):
    """预计算各层 XOR 编码的 gather 索引（与标准极化码变换等价）。"""
    nb_stages = int(np.log2(n))
    ind_gather = np.ones([nb_stages, n + 1], dtype=np.int32) * n
    for s in range(nb_stages):
        ind_range = np.arange(int(n / 2))
        ind_dest = ind_range * 2 - np.mod(ind_range, 2 ** s)
        ind_origin = ind_dest + 2 ** s
        ind_gather[s, ind_dest] = ind_origin
    return ind_gather


def build_generator_matrix(N):
    """G_N = B_N F^{\\otimes n}（GF(2)），用于校验。"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    fn = F.copy()
    n = int(np.log2(N))
    for _ in range(n - 1):
        fn = np.kron(fn, F)
    br = bit_reversal_permutation(N)
    return fn[br] % 2


def polar_encode(u):
    """
    极化码编码。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位，冻结位为 0）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.uint8)
    N = len(u)
    if N == 1:
        return u.copy().astype(int)
    n = int(np.log2(N))
    x = u.copy()
    ind_gather = _gen_xor_indices(N)
    for s in range(n):
        for j in range(N):
            origin = ind_gather[s, j]
            if origin < N:
                x[j] = (x[j] ^ x[origin]) % 2
    return x.astype(int)


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u=", u, "x=", x)
