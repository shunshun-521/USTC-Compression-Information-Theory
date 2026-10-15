"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=np.int64)


def polar_encode(u):
    """
    极化码编码（Arikan 蝶形，与 SC 译码器配套，无额外比特倒序）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    n = len(u)
    stage_len = n
    while stage_len > 1:
        half = stage_len // 2
        for p in range(0, n, stage_len):
            for k in range(half):
                idx = p + k
                u[idx] ^= u[idx + half]
        stage_len = half
    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    # 与 G_4 生成矩阵一致：x = [1,1,0,1]
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
    print("encoder OK:", x)
