"""
极化码 SC（串行抵消）译码器
提供与因子图 BP 一致的 SC（主要实现）及递归形式（调用同一核心）
"""
import math
import numpy as np

LARGE = 1e10


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def _llr_for_bit(phi, llr_ch, frozen_bits, u_partial):
    """计算第 phi 个比特的 LLR（与 BP 因子图左→右消息一致）"""
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    L = np.zeros((N, n + 1), dtype=np.float64)
    R = np.zeros((N, n + 1), dtype=np.float64)
    L[:, n] = llr_ch
    R[frozen_bits, 0] = LARGE
    for i in range(phi):
        R[i, 0] = LARGE if u_partial[i] == 0 else -LARGE

    for j in range(n, 0, -1):
        s = 1 << (j - 1)
        for i in range(0, N, 2 * s):
            for t in range(s):
                idx = i + t
                L[idx, j - 1] = f_operation(R[idx, j] + L[idx + s, j], L[idx, j])
                L[idx + s, j - 1] = f_operation(R[idx, j], L[idx, j]) + L[idx + s, j]

    for j in range(0, n):
        s = 1 << j
        for i in range(0, N, 2 * s):
            for t in range(s):
                idx = i + t
                R[idx, j + 1] = f_operation(R[idx + s, j] + L[idx + s, j + 1], R[idx, j])
                R[idx + s, j + 1] = f_operation(R[idx, j], L[idx, j + 1]) + R[idx + s, j]

    return L[phi, 0] + R[phi, 0]


def sc_decode(llr_ch, frozen_bits):
    """SC 译码（与 BP 因子图一致，适用于比特倒序编码后的 LLR 顺序）"""
    N = len(llr_ch)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    u_hat = np.zeros(N, dtype=np.int64)

    for phi in range(N):
        if frozen_bits[phi]:
            u_hat[phi] = 0
            continue
        metric = _llr_for_bit(phi, llr_ch, frozen_bits, u_hat)
        u_hat[phi] = 0 if metric >= 0 else 1

    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（与 sc_decode 等价，保留接口）"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """保留接口：非递归索引预计算（当前主实现使用因子图 SC）"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = [list(range(n)) for _ in range(N)]
    bit_layer_vec = [[] for _ in range(N)]
    return lambda_offset, llr_layer_vec, bit_layer_vec
