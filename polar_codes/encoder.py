"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def bit_reversed(i, n):
    """单索引比特倒序"""
    result = 0
    for b in range(n):
        if i & (1 << b):
            result |= 1 << (n - 1 - b)
    return result


def polar_encode(u):
    """
    极化码编码：蝶形 XOR（对应 F^{⊗ n}），与 SCD 译码器配套。
    输出经比特倒序以对应 G_N = B_N F^{⊗ n}。
    """
    x = polar_encode_core(u)
    br = bit_reversal_permutation(len(x))
    return x[br].astype(int)


def polar_encode_core(u):
    """蝶形编码核心（无比特倒序）。"""
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    assert 2 ** n == N
    step = 1
    for _ in range(n):
        for i in range(0, N, 2 * step):
            u[i:i + step] ^= u[i + step:i + 2 * step]
        step *= 2
    return u.astype(int)


def polar_encode_no_br(u):
    """别名：蝶形编码（译码器直接使用此域的 LLR）。"""
    return polar_encode_core(u)


def align_llr_for_decoder(llr):
    """
    将信道 LLR 映射到译码器输入域。
    编码输出经比特倒序，故对 LLR 施加相同倒序。
    """
    llr = np.asarray(llr, dtype=np.float64)
    br = bit_reversal_permutation(len(llr))
    return llr[br]
