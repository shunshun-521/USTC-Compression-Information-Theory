"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
Arikan 核 F = [[1,1],[0,1]]，输出比特倒序置换（与标准 PSC 译码器配合）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=np.int64)
    rev = np.zeros(N, dtype=np.int64)
    for i in range(N):
        b = format(i, f"0{n}b")
        rev[i] = int(b[::-1], 2)
    return rev


def polar_encode(u):
    """
    极化码编码（蝶形 + 比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    block = N
    for _ in range(n):
        half = block // 2
        for base in range(0, N, block):
            for k in range(half):
                i = base + k
                u[i] ^= u[i + half]
        block = half

    br = bit_reversal_permutation(N)
    return u[br]


def polar_encode_natural(u):
    """无比特倒序的码字（供 BP 早停重编码等内部使用）"""
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    block = N
    for _ in range(n):
        half = block // 2
        for base in range(0, N, block):
            for k in range(half):
                i = base + k
                u[i] ^= u[i + half]
        block = half
    return u


def channel_llr_to_decoder(llr_ch):
    """将信道 LLR 重排为 PSC 译码器期望的顺序（自然序）"""
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    inv = np.empty(N, dtype=np.int64)
    inv[br] = np.arange(N)
    return llr_ch[inv]


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    # Arikan 核 + 比特倒序：与 u=[1,0,1,1] 对应的标准码字
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"
    print("encoder OK:", x)
