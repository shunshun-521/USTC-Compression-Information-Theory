"""
极化码编码器（非递归蝶形，与 mcba1n polar_encode 一致）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回 rev，满足 out[i] = in[rev[i]]（与 bit_reversed(i,n) 一致）"""
    n = int(np.log2(N))
    return np.array([bit_reversed_int(i, n) for i in range(N)], dtype=np.int64)


def bit_reversed_int(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def polar_encode(u):
    """
    极化码编码：原地蝶形 XOR u[l] ^= u[l + n_split]。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    n = len(u)
    stage_len = n
    while stage_len > 1:
        n_split = stage_len // 2
        for p in range(0, n, stage_len):
            for k in range(n_split):
                l = p + k
                u[l] ^= u[l + n_split]
        stage_len = n_split
    return u


if __name__ == "__main__":
    # 与标准蝶形编码一致；报告中的 N=4 手算向量在另一核定义下给出 [0,0,1,1]
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
    print("encoder test passed:", x)
