"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def remap_channel_llr(llr_ch):
    """将信道 LLR 按比特倒序重排，与编码端 bit-reversal 一致。"""
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[br]


def _sc_decode_core(llr, frozen_bits):
    """SC 译码核心（半分裂递归 + 部分和回传）。"""
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    def decode(llr_ch, frozen_ind):
        n = len(llr_ch)
        if n == 1:
            u = 0 if frozen_ind[0] or llr_ch[0] >= 0 else 1
            return np.array([u], dtype=int), np.array([u], dtype=int)

        half = n // 2
        llr1 = llr_ch[:half]
        llr2 = llr_ch[half:]
        frozen1 = frozen_ind[:half]
        frozen2 = frozen_ind[half:]

        llr_upper = f_operation(llr1, llr2)
        u_hat1, u_hat1_up = decode(llr_upper, frozen1)

        llr_lower = g_operation(llr1, llr2, u_hat1_up)
        u_hat2, u_hat2_up = decode(llr_lower, frozen2)

        u_hat = np.concatenate([u_hat1, u_hat2])
        u_hat1_up = (u_hat1_up ^ u_hat2_up).astype(np.int8)
        u_hat_up = np.concatenate([u_hat1_up, u_hat2_up])
        return u_hat, u_hat_up

    u_hat, _ = decode(llr, frozen_bits)
    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量。"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers_llr = []
        psi = phi
        layer = 0
        while psi % 2 == 1:
            layers_llr.append(layer)
            psi //= 2
            layer += 1
        if layer < n:
            layers_llr.append(layer)
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        if phi % 2 == 0:
            psi = phi
            layer = 0
            while psi % 2 == 0 and psi > 0:
                layers_bit.append(layer)
                psi //= 2
                layer += 1
        else:
            layers_bit.append(0)
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """SC 译码主函数。"""
    llr = remap_channel_llr(llr_ch)
    return _sc_decode_core(llr, frozen_bits)
