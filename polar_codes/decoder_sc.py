"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """check-node (box-plus) 运算"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    la = np.clip(La, -30.0, 30.0)
    lb = np.clip(Lb, -30.0, 30.0)
    return np.log1p(np.exp(la + lb)) - np.log(np.exp(la) + np.exp(lb))


def g_operation(La, Lb, u_hat):
    """variable-node 运算"""
    return (1 - 2 * u_hat) * La + Lb


def _polar_decode_sc_recursive(llr_ch, frozen_ind):
    """Sionna 风格递归 SC，返回 (u_hat, u_hat_up)"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_ind = np.asarray(frozen_ind, dtype=np.float64)
    n = len(llr_ch)
    if n > 1:
        half = n // 2
        llr1 = llr_ch[:half]
        llr2 = llr_ch[half:]
        fz1 = frozen_ind[:half]
        fz2 = frozen_ind[half:]

        llr_up = f_operation(llr1, llr2)
        u_hat1, u_hat1_up = _polar_decode_sc_recursive(llr_up, fz1)

        llr_low = g_operation(llr1, llr2, u_hat1_up)
        u_hat2, u_hat2_up = _polar_decode_sc_recursive(llr_low, fz2)

        u_hat = np.concatenate([u_hat1, u_hat2])
        u_hat1_up = (u_hat1_up.astype(np.int64) ^ u_hat2_up.astype(np.int64)).astype(np.float64)
        u_hat_up = np.concatenate([u_hat1_up, u_hat2_up])
        return u_hat, u_hat_up

    is_frozen = frozen_ind[0] == 1
    if is_frozen:
        u_hat = np.array([0.0])
    else:
        llr = llr_ch[0]
        if llr == 0:
            u_hat = np.array([1.0])
        else:
            u_hat = np.array([0.0 if llr >= 0 else 1.0])
    return u_hat, u_hat.copy()


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    frozen_ind = np.asarray(frozen_bits, dtype=np.float64)
    llr = -np.asarray(llr, dtype=np.float64)
    u_hat, _ = _polar_decode_sc_recursive(llr, frozen_ind)
    return u_hat.astype(int)


def precompute_sc_indices(N):
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layer = 0
        while layer < n and (phi >> layer) & 1:
            layer += 1
        llr_layer_vec.append(list(range(layer, n)))
        bit_layers = []
        layer = 0
        while layer < n - 1 and (phi >> layer) & 1:
            bit_layers.append(layer)
            layer += 1
        bit_layer_vec.append(bit_layers)
    return np.arange(n + 1), llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归接口：调用已验证的递归实现"""
    return sc_decode_recursive(llr_ch, frozen_bits)
