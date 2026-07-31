"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.arange(N)
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int("".join(reversed(format(i, f"0{n}b"))), 2)
    return rev


def polar_encode(u):
    """
    极化码编码（蝶形结构，O(N log N)）。

    采用 XOR 入左半分支的蝶形更新，与 SC 译码器（按比特倒序处理）配套。
    该实现与规范中的「XOR + 比特倒序置换」在信息位重排意义下等价。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    block = N
    for _ in range(int(np.log2(N))):
        half = block // 2
        for p in range(0, N, block):
            for k in range(half):
                u[p + k] ^= u[p + k + half]
        block = half
    return u.astype(int)


if __name__ == "__main__":
    from channel import bpsk_modulate, compute_llr
    from decoder_sc import sc_decode

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u)
    print("x =", x)
    frozen = np.zeros(len(u), dtype=int)
    llr = compute_llr(bpsk_modulate(x), 1e-6)
    u_hat = sc_decode(llr, frozen)
    assert np.array_equal(u_hat, u), f"编码/译码往返错误: {u_hat}"
    print("Encoder round-trip test passed!")
