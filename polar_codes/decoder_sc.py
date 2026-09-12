"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码核心。
    g 运算使用当前层的部分重编码比特 u_hat_up，而非最终 u_hat。
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    if N == 1:
        bit = 0 if frozen_bits[0] or llr[0] >= 0 else 1
        return np.array([bit], dtype=int), np.array([bit], dtype=int)

    half = N // 2
    llr1 = llr[:half]
    llr2 = llr[half:]
    frozen1 = frozen_bits[:half]
    frozen2 = frozen_bits[half:]

    llr_left = f_operation(llr1, llr2)
    u_hat1, u_hat1_up = _sc_decode_recursive(llr_left, frozen1)
    llr_right = g_operation(llr1, llr2, u_hat1_up)
    u_hat2, u_hat2_up = _sc_decode_recursive(llr_right, frozen2)

    u_hat = np.concatenate([u_hat1, u_hat2])
    u_hat1_up_xor = (u_hat1_up ^ u_hat2_up).astype(np.int8)
    u_hat_up = np.concatenate([u_hat1_up_xor, u_hat2_up])
    return u_hat, u_hat_up


def precompute_sc_indices(N):
    """预计算辅助索引（保留接口）"""
    from encoder import bit_reversal_permutation
    return bit_reversal_permutation(N)


def sc_decode(llr_ch, frozen_bits):
    """SC 译码主接口"""
    u_hat, _ = _sc_decode_recursive(llr_ch, frozen_bits)
    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（仅返回 u_hat，供测试脚本使用）"""
    u_hat, _ = _sc_decode_recursive(llr, frozen_bits)
    return u_hat
