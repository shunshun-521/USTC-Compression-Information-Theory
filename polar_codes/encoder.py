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
        b = format(i, f"0{n}b")[::-1]
        rev[i] = int(b, 2)
    return rev


def bit_reversed(i, n):
    """对标量索引 i 做 n 位比特倒序"""
    result = 0
    for k in range(n):
        if i & (1 << k):
            result |= 1 << (n - 1 - k)
    return result


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    蝶形 XOR 编码后做比特倒序置换，与 G_N = B_N F^{⊗n} 一致。
    译码器在比特倒序索引顺序下处理，信道 LLR 保持自然顺序。
    """
    x = polar_encode_butterfly_only(u)
    rev = bit_reversal_permutation(len(x))
    return x[rev]


def polar_encode_butterfly_only(u):
    """仅蝶形 XOR（无比特倒序），供 BP 早停重编码等内部使用"""
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    block = N
    while block > 1:
        half = block // 2
        for start in range(0, N, block):
            for k in range(half):
                idx = start + k
                u[idx] ^= u[idx + half]
        block = half
    return u


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u=", u, "x=", x)
    # 编码器与译码器一致时，端到端仿真通过即可
    from decoder_sc import sc_decode
    from channel import bpsk_modulate, compute_llr

    llr = compute_llr(bpsk_modulate(x), 0.1)
    uh = sc_decode(llr, np.zeros(4, dtype=bool))
    print("roundtrip u_hat", uh)
