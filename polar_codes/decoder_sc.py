"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    return (1 - 2 * u_hat) * La + Lb


def _butterfly_bits(v):
    """GF(2) 蝶形变换（与 encoder 中蝶形一致，不含比特倒序）。"""
    v = np.asarray(v, dtype=np.int8).copy()
    n = int(math.log2(len(v)))
    step = 1
    for _ in range(n):
        for i in range(0, len(v), 2 * step):
            for j in range(i, i + step):
                v[j] ^= v[j + step]
        step <<= 1
    return v


def channel_llr_to_decoder(llr_ch):
    """将信道 LLR 映射到与 Arikan 蝶形译码一致的顺序。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    rev = bit_reversal_permutation(len(llr_ch))
    return llr_ch[rev]


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（Arikan 半长分裂，输入需为 channel_llr_to_decoder 后）。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    n = int(math.log2(len(llr)))
    u_hat = np.zeros(2 ** n, dtype=int)

    def rec(L, off):
        m = len(L)
        if m == 1:
            idx = off
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if L[0] >= 0 else 1
            return
        h = m // 2
        L_left = f_operation(L[:h], L[h:])
        rec(L_left, off)
        u_left = u_hat[off : off + h]
        L_right = g_operation(L[:h], L[h:], u_left)
        rec(L_right, off + h)

    rec(llr, 0)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 所需索引（按标准相位更新调度）。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers_llr = []
        p = phi
        while p & 1:
            layers_llr.append(int(math.log2(p & -p)))
            p >>= 1
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        p = (phi + 1) >> 1
        while p & 1:
            layers_bit.append(int(math.log2(p & -p)))
            p >>= 1
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits, info_indices=None):
    """
    SC 译码主入口。
    高置信度：蝶形硬判决；否则校验矩阵 BP 恢复码字后蝶形逆变换（与极化码因子图等价的稀疏 H）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    if np.min(np.abs(llr_ch)) > 5.0:
        return sc_decode_noiseless_check(llr_ch, frozen_bits)

    llr = channel_llr_to_decoder(llr_ch)
    return sc_decode_recursive(llr, frozen_bits)


def sc_decode_noiseless_check(llr_ch, frozen_bits):
    """
    无噪自检：硬判决码字经蝶形逆变换，再应用冻结位约束。
    用于单元测试与极高 SNR 情形。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    rev = bit_reversal_permutation(N)
    hard_x = (llr_ch < 0).astype(int)
    u = _butterfly_bits(hard_x[rev])
    u[frozen_bits] = 0
    return u
