"""
极化码编码器
编码：u（含冻结位）经极化变换得到码字（与 3GPP / Sionna PolarEncoder 一致）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        r = 0
        for b in range(n):
            if (i >> b) & 1:
                r |= 1 << (n - 1 - b)
        rev[i] = r
    return rev


def _gen_encode_indices(N):
    """预计算 Sionna/3GPP 风格逐层 XOR 的 gather 索引"""
    n = int(np.log2(N))
    ind_gather = np.ones((n, N + 1), dtype=np.int32) * N
    for s in range(n):
        ind_range = np.arange(N // 2)
        ind_dest = ind_range * 2 - np.mod(ind_range, 2**s)
        ind_origin = ind_dest + 2**s
        ind_gather[s, ind_dest] = ind_origin
    return ind_gather


def polar_encode(u):
    """
    极化码编码。
    u: 长度 N 的源向量（信息位 + 冻结位，冻结位为 0）
    """
    u = np.asarray(u, dtype=np.uint8).copy()
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be power of 2")
    n = int(np.log2(N))
    x = np.zeros(N + 1, dtype=np.uint8)
    x[:N] = u
    ind_gather = _gen_encode_indices(N)
    for s in range(n):
        ind_helper = ind_gather[s, :]
        x_add = x[ind_helper]
        x = np.bitwise_xor(x, x_add)
    return x[:N].astype(int)


def polar_generator_matrix(N):
    """构建生成矩阵（用于校验）"""
    G = np.zeros((N, N), dtype=np.int8)
    for i in range(N):
        u = np.zeros(N, dtype=int)
        u[i] = 1
        G[i] = polar_encode(u)
    return G


def polar_encode_matrix(u):
    """矩阵形式编码（校验）"""
    u = np.asarray(u, dtype=np.int8)
    G = polar_generator_matrix(len(u))
    return (u @ G) % 2


def channel_llr_to_decoder(llr_ch):
    """
    信道 LLR 与译码器内部约定对齐：
    本实现使用 LLR = -2y/sigma^2（与 Sionna 2*cw-1 映射一致），无需额外倒序。
    """
    return -np.asarray(llr_ch, dtype=np.float64)


def map_info_to_codeword(info_bits, info_indices, N):
    """将 K 个信息比特映射到长度 N 的 u 向量（冻结位为 0）"""
    u = np.zeros(N, dtype=int)
    u[np.asarray(info_indices, dtype=int)] = info_bits
    return u
