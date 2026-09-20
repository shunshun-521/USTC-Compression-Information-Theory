"""
极化码编码器
编码利用蝶形 XOR 结构，与标准因子图一致（无额外比特倒序）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.uint8).copy()
    N = len(u)
    n = int(np.log2(N))
    for s in range(n):
        ind_range = np.arange(N // 2)
        ind_dest = ind_range * 2 - np.mod(ind_range, 2 ** s)
        ind_origin = ind_dest + 2 ** s
        u[ind_dest] = np.bitwise_xor(u[ind_dest], u[ind_origin])
    return u.astype(np.int8)


def build_generator_matrix(N):
    """构造生成矩阵（用于验证）"""
    cols = []
    for i in range(N):
        u = np.zeros(N, dtype=np.int8)
        u[i] = 1
        cols.append(polar_encode(u))
    return np.array(cols).T


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u:", u, "x:", x)
