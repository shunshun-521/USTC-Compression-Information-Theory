"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def _bit_reverse_indices(N):
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    return _bit_reverse_indices(N)


def _butterfly_encode(u):
    """u * F^{\\otimes n}（无输出比特倒序）。"""
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    step = 1
    for _ in range(n):
        for i in range(0, N, 2 * step):
            u[i:i + step] ^= u[i + step:i + 2 * step]
        step *= 2
    return u


def polar_encode(u, apply_output_bit_reversal=False):
    """
    极化码编码。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）
        apply_output_bit_reversal: 若为 True，对码字做比特倒序（x = u * B_N * F^{\\otimes n}）

    返回：
        x: 长度为 N 的码字
    """
    x = _butterfly_encode(u)
    if apply_output_bit_reversal:
        br = _bit_reverse_indices(len(x))
        x = x[br]
    return x.astype(int)


if __name__ == "__main__":
    u = np.array([0, 0, 1, 1])
    x = polar_encode(u)
    print("encode test (F^\\otimes n):", x)
    assert np.array_equal(x, [0, 1, 0, 1]), f"编码器错误: {x}"
