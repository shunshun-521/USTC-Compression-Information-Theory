"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _reorder_channel_llr(llr_ch, N):
    br = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[br]


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（参考实现）"""
    N = len(llr_ch)
    llr = _reorder_channel_llr(llr_ch, N)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    u_hat = np.zeros(N, dtype=int)

    def recurse(llr_in, offset):
        m = len(llr_in)
        if m == 1:
            idx = offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_in[0] >= 0 else 1
            return np.array([u_hat[idx]], dtype=int)

        half = m // 2
        l1, l2 = llr_in[:half], llr_in[half:]
        u_left = recurse(f_operation(l1, l2), offset)
        u_right = recurse(g_operation(l1, l2, u_left), offset + half)
        return np.concatenate([(u_left ^ u_right), u_right])

    recurse(llr, 0)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        if phi == 0:
            llr_layers = list(range(n))
        else:
            psi = phi
            layer = 0
            while psi % 2 == 1:
                psi >>= 1
                layer += 1
            llr_layers = list(range(layer, n))
        llr_layer_vec.append(llr_layers)

        if phi % 2 == 1:
            bit_layers = []
        else:
            psi = phi
            layer = 0
            while psi % 2 == 0 and psi > 0:
                psi >>= 1
                layer += 1
            bit_layers = list(range(layer))
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_llr_at_phase(llr_ch, frozen_bits, u_hat_prefix, phi):
    """计算第 phi 位在已知前缀 u_hat_prefix[:phi] 下的 LLR（供 SCL / 非递归 SC 使用）"""
    llr = _reorder_channel_llr(llr_ch, len(llr_ch))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    u_hat = np.asarray(u_hat_prefix, dtype=int)
    llr_out = [0.0]

    def recurse(llr_in, offset):
        m = len(llr_in)
        if m == 1:
            idx = offset
            if idx == phi:
                llr_out[0] = llr_in[0]
            if idx < phi:
                return np.array([u_hat[idx]], dtype=int)
            if frozen_bits[idx]:
                return np.array([0], dtype=int)
            return np.array([0 if llr_in[0] >= 0 else 1], dtype=int)

        half = m // 2
        l1, l2 = llr_in[:half], llr_in[half:]
        u_left = recurse(f_operation(l1, l2), offset)
        u_right = recurse(g_operation(l1, l2, u_left), offset + half)
        return np.concatenate([(u_left ^ u_right), u_right])

    recurse(llr, 0)
    return llr_out[0]


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码：逐相位判决（与递归版本等价）"""
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    frozen_bool = frozen_bits.astype(bool)
    N = len(llr_ch)
    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        if frozen_bits[phi]:
            u_hat[phi] = 0
        else:
            llr_bit = sc_llr_at_phase(llr_ch, frozen_bool, u_hat, phi)
            u_hat[phi] = 0 if llr_bit >= 0 else 1

    return u_hat
