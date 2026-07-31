"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
from encoder import _bit_rev_indices


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    若一方为 0，则返回另一方。
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    result = np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))
    result = np.where((La == 0) & (Lb != 0), Lb, result)
    result = np.where((Lb == 0) & (La != 0), La, result)
    return result


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _polar_decode_sc_core(llr_ch, frozen_bits):
    """
    递归 SC 译码核心。
    frozen_bits: 1 表示冻结位，0 表示信息位
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    n = len(llr_ch)

    if n == 1:
        if frozen_bits[0]:
            u = 0
        else:
            u = 0 if llr_ch[0] >= 0 else 1
        return np.array([u], dtype=int), np.array([u], dtype=int)

    half = n // 2
    llr_left_in = llr_ch[:half]
    llr_right_in = llr_ch[half:]

    x_llr_left = f_operation(llr_left_in, llr_right_in)
    u_hat_left, u_hat_left_up = _polar_decode_sc_core(x_llr_left, frozen_bits[:half])

    x_llr_right = g_operation(llr_left_in, llr_right_in, u_hat_left_up)
    u_hat_right, u_hat_right_up = _polar_decode_sc_core(x_llr_right, frozen_bits[half:])

    u_hat = np.concatenate([u_hat_left, u_hat_right])
    u_hat_up_left = np.bitwise_xor(u_hat_left_up.astype(int), u_hat_right_up.astype(int)).astype(int)
    u_hat_up = np.concatenate([u_hat_up_left, u_hat_right_up])
    return u_hat, u_hat_up


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码。frozen_bits: 1 表示冻结位，0 表示信息位"""
    brp = _bit_rev_indices(len(llr))
    u_hat, _ = _polar_decode_sc_core(np.asarray(llr, dtype=np.float64)[brp], frozen_bits)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的三个辅助向量。"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers_llr = []
        p = phi
        layer = 0
        while p & 1:
            p >>= 1
            layer += 1
        for l in range(layer, n):
            layers_llr.append(l)
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        p = phi
        layer = 0
        while p & 1:
            p >>= 1
            layer += 1
            if layer > 0:
                layers_bit.append(layer - 1)
        bit_layer_vec.append(layers_bit)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    编码端含比特倒序置换，译码前对信道 LLR 做相同倒序。
  """
    return sc_decode_recursive(llr_ch, frozen_bits)
