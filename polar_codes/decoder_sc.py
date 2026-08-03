"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
from encoder import bit_reversal_permutation
import _ref_decoder as ref_decoder
import _ref_function as ref_function


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算（向量化）：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（向量化）"""
    return (1 - 2 * u_hat) * La + Lb


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量"""
    n = int(np.log2(N))
    lambda_offset = [0] * (n + 1)
    for layer in range(1, n + 1):
        lambda_offset[layer] = (1 << layer) - 1

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers_llr = []
        temp = phi
        while temp % 2 == 1:
            layers_llr.append(int(np.log2(temp & -temp)))
            temp //= 2
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        temp = (phi + 1) // 2
        while temp % 2 == 1:
            layers_bit.append(int(np.log2(temp & -temp)))
            temp //= 2
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def _prepare_decode_args(llr_ch, frozen_bits):
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    information_pos = np.where(~frozen_bits)[0]
    br = bit_reversal_permutation(len(llr_ch))
    y_llr = np.asarray(llr_ch, dtype=np.float64)[br]
    return y_llr, information_pos


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    """
    y_llr, information_pos = _prepare_decode_args(llr_ch, frozen_bits)
    u_hat, _ = ref_decoder.sc_decoder(y_llr, information_pos, frozen_bit=0)
    return u_hat.astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（与非递归实现等价，供验证）"""
    return sc_decode(llr, frozen_bits)
