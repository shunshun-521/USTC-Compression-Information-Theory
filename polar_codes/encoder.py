"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(format(i, f'0{n}b')[::-1], 2)
    return rev


def _gen_encode_indices(N):
    """预计算编码阶段的 gather 索引（Sionna/Arikan 风格）"""
    n_stages = int(np.log2(N))
    ind_gather = np.ones((n_stages, N + 1), dtype=np.int32) * N
    for s in range(n_stages):
        ind_range = np.arange(N // 2)
        ind_dest = ind_range * 2 - np.mod(ind_range, 2 ** s)
        ind_origin = ind_dest + 2 ** s
        ind_gather[s, ind_dest] = ind_origin
    return ind_gather


def polar_encode(u):
    """
    极化码编码（Sionna/Arikan 蝶形 XOR 结构）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.array(u, dtype=np.uint8).copy()
    N = len(u)
    n = int(np.log2(N))
    ind_gather = _gen_encode_indices(N)

    x = np.zeros(N + 1, dtype=np.uint8)
    x[:N] = u

    for s in range(n):
        ind_helper = ind_gather[s]
        x = np.bitwise_xor(x, x[ind_helper])

    return x[:N].astype(int)


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print(f"u={u} -> x={x}")
