"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f'{i:0{n}b}'[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字

    实现：蝶形（butterfly）递归结构
        - 每层：u[l] ^= u[l + step]（左半累加右半）
        - 共 log2(N) 层
        - 最后做比特倒序置换（bit-reversal permutation）
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    block = N
    for _ in range(n):
        n_split = block // 2
        if n_split == 0:
            break
        for p in range(0, N, block):
            for k in range(n_split):
                u[p + k] ^= u[p + k + n_split]
        block = n_split
    br = bit_reversal_permutation(N)
    return u[br]


if __name__ == '__main__':
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print(f"u={u} -> x={x}")
    # G_N @ u（Arikan 构造，列向量约定）对应 x=[1,0,1,1]
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"
    print("Encoder test passed.")
