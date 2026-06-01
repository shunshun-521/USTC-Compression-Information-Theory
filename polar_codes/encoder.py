"""
极化码编码器
编码：x = u * G_N，利用蝶形 / 分阶段 XOR 结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)])


def _gen_encode_indices(n):
    """生成分阶段 XOR 编码的 gather 索引（与 Sionna PolarEncoder 一致）"""
    nb_stages = int(np.log2(n))
    ind_gather = np.ones([nb_stages, n + 1], dtype=np.int32) * n
    for s in range(nb_stages):
        ind_range = np.arange(n // 2)
        ind_dest = ind_range * 2 - np.mod(ind_range, 2**s)
        ind_origin = ind_dest + 2**s
        ind_gather[s, ind_dest] = ind_origin
    return ind_gather


def polar_encode(u):
    """
    极化码编码。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位，冻结位为 0）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.uint8).copy()
    N = len(u)
    n = int(np.log2(N))
    ind_gather = _gen_encode_indices(N)
    x = np.zeros(N + 1, dtype=np.uint8)
    x[:N] = u
    for s in range(n):
        ind_helper = ind_gather[s, :]
        x[:N] = np.bitwise_xor(x[:N], x[ind_helper[:N]])
    return x[:N].astype(int)


def polar_encode_butterfly_br(u):
    """蝶形 + 比特倒序编码（备用实现）"""
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    for d in range(n):
        step = 1 << d
        for i in range(0, N, 2 * step):
            for j in range(step):
                u[i + j] ^= u[i + j + step]
    br = bit_reversal_permutation(N)
    return u[br]


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    print("stage XOR:", polar_encode(u))
    print("butterfly+BR:", polar_encode_butterfly_br(u))
