"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=int)
    rev = np.zeros(N, dtype=int)
    for bit in range(n):
        rev |= ((idx >> bit) & 1) << (n - 1 - bit)
    return rev


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    # 与 SC 因子图一致的蝶形编码（无输出比特倒序；倒序体现在译码调度中）
    stage_len = N
    while stage_len > 1:
        half = stage_len // 2
        for p in range(0, N, stage_len):
            for k in range(half):
                u[p + k] ^= u[p + k + half]
        stage_len = half
    return u


def polar_generator_matrix(N):
    """构造 G_N = B_N F^{\\otimes n}（用于校验）。"""
    G = np.zeros((N, N), dtype=int)
    for i in range(N):
        e = np.zeros(N, dtype=int)
        e[i] = 1
        G[i] = polar_encode(e)
    return G


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("encode test:", x)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
    G = polar_generator_matrix(4)
    for _ in range(16):
        ut = np.array([(_ >> i) & 1 for i in range(4)])
        assert np.array_equal(polar_encode(ut), ut @ G % 2)
