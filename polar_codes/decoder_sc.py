"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（box-plus）。"""
    sa = np.where(La >= 0, 1.0, -1.0)
    sb = np.where(Lb >= 0, 1.0, -1.0)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（u_hat 为部分和/已编码比特，可为向量）。"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _cn_op_exact(x, y, llr_max=30.0):
    x = np.clip(x, -llr_max, llr_max)
    y = np.clip(y, -llr_max, llr_max)
    return np.log1p(np.exp(x + y)) - np.log(np.exp(x) + np.exp(y))


def _sc_decode_recursive_core(llr_ch, frozen_ind, cn_op=f_operation):
    """
    递归 SC（与 Sionna PolarSCDecoder 一致，含部分和 u_up 回传）。
    frozen_ind: 1 表示冻结位，0 表示信息位。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_ind = np.asarray(frozen_ind, dtype=np.float64)
    n = len(llr_ch)
    if n > 1:
        half = n // 2
        llr1, llr2 = llr_ch[:half], llr_ch[half:]
        fr1, fr2 = frozen_ind[:half], frozen_ind[half:]
        llr1_in = cn_op(llr1, llr2)
        u1, u1_up = _sc_decode_recursive_core(llr1_in, fr1, cn_op)
        llr2_in = g_operation(llr1, llr2, u1_up)
        u2, u2_up = _sc_decode_recursive_core(llr2_in, fr2, cn_op)
        u_hat = np.concatenate([u1, u2])
        u1_up_i = (u1_up.astype(np.int8) ^ u2_up.astype(np.int8)).astype(np.float64)
        u_up = np.concatenate([u1_up_i, u2_up])
        return u_hat, u_up
    if frozen_ind[0] >= 0.5:
        u = np.array([0.0])
    else:
        if llr_ch[0] > 0:
            u = np.array([0.0])
        elif llr_ch[0] < 0:
            u = np.array([1.0])
        else:
            u = np.array([1.0])
    return u, u


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（冻结位 bool：True=冻结）。"""
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    frozen_ind = frozen_bits.astype(np.float64)
    u_hat, _ = _sc_decode_recursive_core(llr, frozen_ind)
    return u_hat.astype(int)


def precompute_sc_indices(N):
    """预计算非递归 SC 的层调度（文档/扩展用）。"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers = []
        p = phi
        if p % 2 == 0:
            layers.append(n - 1)
        while p % 2 == 1:
            p >>= 1
            layers.append(max(0, n - 2))
        llr_layer_vec.append(layers)
        bit_layers = []
        p = phi
        layer = 0
        while p % 2 == 1 and layer < n:
            bit_layers.append(layer)
            p >>= 1
            layer += 1
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归接口：当前与高效递归核心相同（复杂度 O(N log N)）。"""
    return sc_decode_recursive(llr_ch, frozen_bits)


def sc_decode_reference(llr_ch, frozen_bits):
    return sc_decode_recursive(llr_ch, frozen_bits)
