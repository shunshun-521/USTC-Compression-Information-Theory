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
        b = format(i, f"0{n}b")
        rev[i] = int(b[::-1], 2)
    return rev


def polar_encode(u):
    """
    极化码编码（Arikan 蝶形：每层 [a^b, b] 合并）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    assert 2**n == N

    m = 1
    for _ in range(n):
        for i in range(0, N, 2 * m):
            x_part = u[i : i + m]
            y_part = u[i + m : i + 2 * m]
            u[i : i + 2 * m] = np.concatenate([(x_part ^ y_part) % 2, y_part])
        m *= 2
    return u


def prepare_decoder_llr(llr_ch):
    """信道 LLR 预处理（恒等映射）"""
    return np.asarray(llr_ch, dtype=np.float64)
