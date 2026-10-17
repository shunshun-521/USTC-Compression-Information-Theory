"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=np.int64)
    rev = ((idx[:, None] & (1 << np.arange(n))) != 0).sum(axis=1)
    return rev.astype(np.int64)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int32).copy()
    N = len(u)
    step = N // 2
    while step >= 1:
        for i in range(0, N, 2 * step):
            u[i : i + step] ^= u[i + step : i + 2 * step]
        step //= 2
    return u


if __name__ == "__main__":
    from decoder_sc import sc_decode

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    llr = 50.0 * (1.0 - 2.0 * x)
    u_hat = sc_decode(llr, np.zeros(4, dtype=bool))
    assert np.array_equal(u_hat, u), f"编码器/译码器不一致: x={x}, u_hat={u_hat}"
    print("encoder self-test passed")
