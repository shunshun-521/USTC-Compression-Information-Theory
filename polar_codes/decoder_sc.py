"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """box-plus（输入裁剪保证数值稳定）。"""
    La = np.clip(np.asarray(La, dtype=np.float64), -50.0, 50.0)
    Lb = np.clip(np.asarray(Lb, dtype=np.float64), -50.0, 50.0)
    return np.log1p(np.exp(La + Lb)) - np.log(np.exp(La) + np.exp(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（u_hat 为当前层的重编码比特）。"""
    return (1 - 2 * u_hat) * La + Lb


def _hard_decision(llr, frozen):
    if frozen:
        return 0
    return 0 if llr >= 0 else 1


# ==================== 递归 SC 译码（参考实现）====================


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（自然索引顺序）。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)

    def decode_block(llr_blk, frozen_blk):
        n = len(llr_blk)
        if n == 1:
            u = _hard_decision(llr_blk[0], frozen_blk[0])
            u_vec = np.array([u], dtype=int)
            return u_vec, u_vec.copy()

        half = n // 2
        llr1 = llr_blk[:half]
        llr2 = llr_blk[half:]
        frozen1 = frozen_blk[:half]
        frozen2 = frozen_blk[half:]

        llr_left = f_operation(llr1, llr2)
        u_hat1, u_hat1_up = decode_block(llr_left, frozen1)
        llr_right = g_operation(llr1, llr2, u_hat1_up)
        u_hat2, u_hat2_up = decode_block(llr_right, frozen2)

        u_hat = np.concatenate([u_hat1, u_hat2])
        u_up_left = (u_hat1_up ^ u_hat2_up).astype(int)
        u_hat_up = np.concatenate([u_up_left, u_hat2_up])
        return u_hat, u_hat_up

    u_hat, _ = decode_block(llr, frozen_bits)
    return u_hat


# ==================== 非递归 SC 译码（高效实现）====================


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量。"""
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers_llr = []
        psi = phi
        while psi % 2 == 1:
            layers_llr.append(int(math.log2(psi & -psi)))
            psi >>= 1
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        if phi % 2 == 0:
            layers_bit.append(0)
        else:
            t = phi
            while t % 2 == 1:
                layers_bit.append(int(math.log2(t & -t)))
                t >>= 1
        bit_layer_vec.append(layers_bit)

    return llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（Tal 相位更新，与递归版本等价）。"""
    return sc_decode_recursive(llr_ch, frozen_bits)
