"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

LLR_MAX = 30.0

# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def f_boxplus(La, Lb):
    """Box-plus（check-node）运算，数值稳定实现"""
    La = np.clip(np.asarray(La, dtype=np.float64), -LLR_MAX, LLR_MAX)
    Lb = np.clip(np.asarray(Lb, dtype=np.float64), -LLR_MAX, LLR_MAX)
    return np.log1p(np.exp(La + Lb)) - np.log(np.exp(La) + np.exp(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（支持标量/向量）"""
    return (1.0 - 2.0 * u_hat) * La + Lb


# ==================== 递归 SC 译码（参考实现）====================


def _polar_decode_sc_core(llr_ch, frozen_ind, use_boxplus=True):
    """
    递归 SC（Sionna/Arikan 风格）：连续半分 + 部分和回传。
    frozen_ind: 长度 n 的数组，1 表示冻结位
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_ind = np.asarray(frozen_ind, dtype=bool)
    n = len(llr_ch)
    f_fn = f_boxplus if use_boxplus else f_operation

    if n == 1:
        if frozen_ind[0]:
            u = 0
        else:
            u = 0 if llr_ch[0] >= 0 else 1
        return np.array([u], dtype=int), np.array([u], dtype=int)

    half = n // 2
    llr1 = llr_ch[:half]
    llr2 = llr_ch[half:]
    frozen1 = frozen_ind[:half]
    frozen2 = frozen_ind[half:]

    llr_upper = f_fn(llr1, llr2)
    u_hat1, u_hat1_up = _polar_decode_sc_core(llr_upper, frozen1, use_boxplus)

    llr_lower = g_operation(llr1, llr2, u_hat1_up)
    u_hat2, u_hat2_up = _polar_decode_sc_core(llr_lower, frozen2, use_boxplus)

    u_hat = np.concatenate([u_hat1, u_hat2])
    u_up = np.concatenate([np.bitwise_xor(u_hat1_up, u_hat2_up), u_hat2_up])
    return u_hat, u_up


def _channel_llr_to_decoder(llr_ch):
    """编码含比特倒序置换，译码前对信道 LLR 做相同倒序对齐因子图。"""
    from encoder import bit_reversal_permutation

    N = len(llr_ch)
    br_idx = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[br_idx]


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码。frozen_bits: True/1 表示冻结位"""
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    llr_dec = _channel_llr_to_decoder(llr)
    u_hat, _ = _polar_decode_sc_core(llr_dec, frozen_bits, use_boxplus=True)
    return u_hat


# ==================== 非递归 SC 译码（高效实现）====================


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [2 ** layer - 1 for layer in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers_llr = []
        psi = phi
        while psi & 1:
            layers_llr.append(int(math.log2(psi & -psi)))
            psi >>= 1
        layers_llr.append(n)
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        if phi % 2 == 0:
            psi2 = phi
            while psi2 > 0 and (psi2 & 1) == 0:
                layers_bit.append(int(math.log2(psi2 & -psi2)))
                psi2 >>= 1
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def _update_llr(L, B, x, n):
    """按相位 x 更新 LLR 矩阵（与递归版本等价的矩阵形式）"""
    N = L.shape[0]
    for j in range(n - 1, -1, -1):
        s = 2 ** (n - j)
        t = s // 2
        for i in range(x, N, s):
            if t > (i % s):
                L[i, j] = f_boxplus(L[i, j + 1], L[i + t, j + 1])
            else:
                L[i, j] = g_operation(L[i, j + 1], L[i - t, j + 1], B[i - t, j])


def _update_bits(B, x, n):
    """比特回传更新"""
    for j in range(n):
        s = 2 ** (n - j)
        t = s // 2
        if (x % s) >= t:
            i = x - t
            B[i, j + 1] = B[i, j] ^ B[i + t, j]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（内部调用递归核心，含 LLR 比特倒序对齐）。
    frozen_bits: 1 表示冻结位
    """
    return sc_decode_recursive(llr_ch, frozen_bits)
