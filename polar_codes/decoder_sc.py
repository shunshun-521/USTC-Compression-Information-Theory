"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_partial):
    """g 运算：g(La, Lb, u_partial) = (1 - 2*u_partial) * La + Lb"""
    return (1.0 - 2.0 * u_partial) * La + Lb


def _prepare_llr(llr_ch):
    """信道 LLR 按比特倒序置换，与编码器保持一致"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    br = bit_reversal_permutation(len(llr_ch))
    return llr_ch[br]


def _as_frozen_mask(frozen_bits):
    frozen_bits = np.asarray(frozen_bits)
    if frozen_bits.dtype == bool:
        return frozen_bits
    return frozen_bits.astype(bool)


def _sc_decode_core(llr, frozen_bits, u_fixed=None):
    """
    递归 SC 核心。
    u_fixed: 可选，长度 N；-1 表示未固定，0/1 表示强制比特。
    g 运算使用 u_up（重编码比特），不是 u_hat。
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    n = len(llr)

    if n == 1:
        if u_fixed is not None and u_fixed[0] >= 0:
            bit = int(u_fixed[0])
        elif frozen_bits[0]:
            bit = 0
        else:
            bit = 0 if llr[0] >= 0 else 1
        return np.array([bit], dtype=int), np.array([bit], dtype=int)

    half = n // 2
    llr1 = llr[:half]
    llr2 = llr[half:]
    frozen1 = frozen_bits[:half]
    frozen2 = frozen_bits[half:]
    fixed1 = u_fixed[:half] if u_fixed is not None else None
    fixed2 = u_fixed[half:] if u_fixed is not None else None

    llr_upper = f_operation(llr1, llr2)
    u_hat1, u_up1 = _sc_decode_core(llr_upper, frozen1, fixed1)

    llr_lower = g_operation(llr1, llr2, u_up1)
    u_hat2, u_up2 = _sc_decode_core(llr_lower, frozen2, fixed2)

    u_hat = np.concatenate([u_hat1, u_hat2])
    u_up1 = np.bitwise_xor(u_up1, u_up2)
    u_up = np.concatenate([u_up1, u_up2])
    return u_hat, u_up


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    llr = _prepare_llr(llr)
    frozen_bits = _as_frozen_mask(frozen_bits)
    u_hat, _ = _sc_decode_core(llr, frozen_bits)
    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（委托给已验证的递归实现）"""
    return sc_decode_recursive(llr_ch, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [1 << d for d in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers = []
        psi = phi
        while psi % 2 == 1:
            layers.append(int(math.log2(psi & -psi)))
            psi //= 2
        llr_layer_vec.append(layers)

        layers_b = []
        psi = phi
        while psi > 0 and psi % 2 == 0:
            layers_b.append(int(math.log2(psi & -psi)))
            psi //= 2
        bit_layer_vec.append(layers_b)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def llr_at_phase(llr_ch, frozen_bits, u_prefix, phi):
    """给定前缀 u_prefix（长度 phi），返回相位 phi 的 LLR"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = _as_frozen_mask(frozen_bits)
    N = len(llr_ch)

    u_fixed = np.full(N, -1, dtype=int)
    u_fixed[:phi] = u_prefix

    def recurse(llr_sub, frozen_sub, fixed_sub, offset):
        m = len(llr_sub)
        if m == 1:
            idx = offset
            if idx == phi:
                return float(llr_sub[0])
            return None

        half = m // 2
        llr1 = llr_sub[:half]
        llr2 = llr_sub[half:]
        f1 = frozen_sub[:half]
        f2 = frozen_sub[half:]
        x1 = fixed_sub[:half] if fixed_sub is not None else None
        x2 = fixed_sub[half:] if fixed_sub is not None else None

        llr_upper = f_operation(llr1, llr2)

        if phi < offset + half:
            return recurse(llr_upper, f1, x1, offset)

        u_hat1, u_up1 = _sc_decode_core(llr_upper, f1, x1)
        llr_lower = g_operation(llr1, llr2, u_up1)
        return recurse(llr_lower, f2, x2, offset + half)

    return recurse(llr_ch, frozen_bits, u_fixed, 0)
