"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1 - 2 * np.asarray(u_hat)) * La + Lb


def _depermute_llr(llr_ch):
    """将信道 LLR 调整为译码树自然顺序（与编码端比特倒序对应）。"""
    N = len(llr_ch)
    return llr_ch[bit_reversal_permutation(N)]


def _sc_decode_recursive_core(llr, frozen_bits):
    """递归 SC 译码核心（输入已去置换）。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    n = len(llr)

    if n == 1:
        if frozen_bits[0]:
            bit = 0
        else:
            bit = 0 if llr[0] >= 0 else 1
        return np.array([bit], dtype=int), np.array([bit], dtype=int)

    half = n // 2
    llr_left = f_operation(llr[:half], llr[half:])
    u_left, u_left_up = _sc_decode_recursive_core(llr_left, frozen_bits[:half])
    llr_right = g_operation(llr[:half], llr[half:], u_left_up)
    u_right, u_right_up = _sc_decode_recursive_core(llr_right, frozen_bits[half:])

    u_hat = np.concatenate([u_left, u_right])
    u_up = np.concatenate([
        (u_left_up.astype(int) ^ u_right_up.astype(int)).astype(int),
        u_right_up.astype(int),
    ])
    return u_hat, u_up


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现）。
    """
    llr = _depermute_llr(np.asarray(llr, dtype=np.float64))
    u_hat, _ = _sc_decode_recursive_core(llr, frozen_bits)
    return u_hat


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    """
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        llr_layers = []
        tmp = phi
        layer = 0
        while tmp % 2 == 1 and layer < n:
            llr_layers.append(layer)
            tmp //= 2
            layer += 1
        llr_layers.append(min(layer, n - 1))
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        tmp = (phi + 1) // 2
        layer = 0
        while tmp % 2 == 1 and layer < n:
            bit_layers.append(layer)
            tmp //= 2
            layer += 1
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    """
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    llr = _depermute_llr(np.asarray(llr_ch, dtype=np.float64))

    try:
        from decoder_sc_fast import sc_decode_numba
        fast = sc_decode_numba(llr, frozen_bits)
        if fast is not None:
            return fast
    except ImportError:
        pass

    u_hat, _ = _sc_decode_recursive_core(llr, frozen_bits)
    return u_hat
