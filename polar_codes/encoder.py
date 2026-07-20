"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def _butterfly_encode(u):
    """蝶形 XOR（左半累加右半），与极化码标准生成矩阵一致。"""
    u = np.asarray(u, dtype=np.int8).copy()
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


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字

    实现：
        1. 蝶形 XOR 编码
        2. 比特倒序置换：x[i] = encoded[br(i)]
    """
    encoded = _butterfly_encode(u)
    br = bit_reversal_permutation(len(encoded))
    return encoded[br].astype(int)


def polar_encode_natural(u):
    """无输出比特倒序的编码（与 natural-order LLR 译码配对）。"""
    return _butterfly_encode(u).astype(int)


def channel_llr_to_decoder(llr_ch):
    """将信道 LLR 转换为译码器所需的比特倒序顺序。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    br = bit_reversal_permutation(len(llr_ch))
    return llr_ch[br]
