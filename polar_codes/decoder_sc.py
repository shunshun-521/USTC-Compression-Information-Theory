"""
极化码 SC（串行抵消）译码器
"""
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat)
    return (1 - 2 * u_hat) * La + Lb


def _decode_block(y, s, frozen):
    m = len(y)
    if m == 1:
        i = s
        return np.array([0 if frozen[i] else (0 if y[0] >= 0 else 1)], dtype=int)
    m2 = m // 2
    ya, yb = y[:m2], y[m2:]
    yl = f_operation(ya, yb)
    ul = _decode_block(yl, s, frozen)
    yr = g_operation(ya, yb, ul)
    ur = _decode_block(yr, s + m2, frozen)
    return np.concatenate([ul, ur])


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    return _decode_block(llr, 0, frozen_bits)


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC：对信道 LLR 做比特倒序后进入因子图，输出自然序源比特估计。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    llr_tree = llr_ch[br]
    return sc_decode_recursive(llr_tree, frozen_bits)


def precompute_sc_indices(N):
    n = int(np.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers = []
        p = phi
        while p & 1:
            layers.append(len(layers))
            p >>= 1
        llr_layer_vec.append(layers)
        bit_layers = layers.copy()
        if phi != N - 1:
            bit_layers.append(len(bit_layers))
        bit_layer_vec.append(bit_layers)
    lambda_offset = [1 << i for i in range(n + 1)]
    return lambda_offset, llr_layer_vec, bit_layer_vec
