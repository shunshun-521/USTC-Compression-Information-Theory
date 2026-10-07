"""
极化码 SC（串行抵消）译码器
"""
import numpy as np
import math

from encoder import bit_reversal_permutation


def _sgn(x):
    x = np.asarray(x, dtype=np.float64)
    return (x > 0).astype(np.float64) - (x < 0).astype(np.float64)


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（box-plus）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    abs_a = np.abs(La)
    abs_b = np.abs(Lb)
    sgn_ab = _sgn(La) * _sgn(Lb)
    return np.where(abs_a >= abs_b, sgn_ab * abs_b, sgn_ab * abs_a)


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    u_hat = np.asarray(u_hat)
    return (1.0 - 2.0 * u_hat) * La + Lb


def _frozen_bool(frozen_bits):
    fb = np.asarray(frozen_bits)
    if fb.dtype == bool:
        return fb
    return fb.astype(np.int8) != 0


def _sc_decode_core(llr_ch, frozen_bits):
    """
    块递归 SC。编码满足 x = (u @ F^{⊗n})[bit_rev]，
    故使用 llr_std = llr_ch[bit_rev] 后按 F^{⊗n} 树译码。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    llr = llr_ch[br]
    frozen_bits = _frozen_bool(frozen_bits)

    u_hat = np.zeros(N, dtype=int)

    def decode_blk(L, offset):
        m = len(L)
        if m == 1:
            i = offset
            if frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if L[0] >= 0 else 1
            return
        h = m // 2
        L_left = f_operation(L[:h], L[h:])
        decode_blk(L_left, offset)
        L_right = g_operation(L[:h], L[h:], u_hat[offset : offset + h])
        decode_blk(L_right, offset + h)

    decode_blk(llr, 0)
    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    return _sc_decode_core(llr, frozen_bits)


def precompute_sc_indices(N):
    n = int(math.log2(N))
    lambda_offset = [1 << s for s in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers = []
        p = phi
        s = 0
        while (p & 1) == 1 and s < n:
            layers.append(s)
            p >>= 1
            s += 1
        llr_layer_vec.append(layers)
        bit_layer_vec.append(layers.copy())
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    return _sc_decode_core(llr_ch, frozen_bits)
