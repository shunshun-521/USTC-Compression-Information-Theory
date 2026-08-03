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
        b = format(i, f'0{n}b')[::-1]
        rev[i] = int(b, 2)
    return rev


def bit_reversed(i, n):
    """单索引比特倒序"""
    result = 0
    for bit in range(n):
        if i & (1 << bit):
            result |= 1 << (n - 1 - bit)
    return result


def polar_encode(u):
    """
    极化码编码（蝶形递归结构，与 SC 译码器配套）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    _polar_encode_recursive(u, 0, N - 1)
    return u


def _polar_encode_recursive(u, i1, i2):
    """递归蝶形编码：左半部分与右半部分异或。"""
    h_shift = (i2 - i1 + 1) // 2
    if h_shift < 1:
        return
    mid = i1 + h_shift
    for k in range(i1, mid):
        u[k] = u[k] ^ u[k + h_shift]
    if h_shift >= 2:
        _polar_encode_recursive(u, i1, mid - 1)
        _polar_encode_recursive(u, mid, i2)


if __name__ == "__main__":
    from decoder_sc import sc_decode
    from channel import bpsk_modulate, compute_llr

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u, "-> x =", x)
    frozen = np.zeros(len(u), dtype=bool)
    llr = compute_llr(bpsk_modulate(x), 0.01)
    u_hat = sc_decode(llr, frozen)
    assert np.array_equal(u_hat, u), f"编码器/译码器不一致: {u_hat}"
    print("Encoder round-trip test passed.")
