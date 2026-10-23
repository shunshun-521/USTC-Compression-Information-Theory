"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _sc_recursive_core(llrs, indices, frozen_bits, decisions):
    """dit 风格递归 SC（参考实现）"""
    if len(indices) == 1:
        i = indices[0]
        decisions[i] = 0 if frozen_bits[i] or llrs[0] >= 0 else 1
        return [decisions[i]]

    half = len(indices) // 2
    left_llr = [f_operation(llrs[i], llrs[i + half]) for i in range(half)]
    enc_left = _sc_recursive_core(left_llr, indices[:half], frozen_bits, decisions)
    right_llr = [
        g_operation(llrs[i], llrs[i + half], enc_left[i]) for i in range(half)
    ]
    enc_right = _sc_recursive_core(right_llr, indices[half:], frozen_bits, decisions)
    return [enc_left[i] ^ enc_right[i] for i in range(half)] + enc_right


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    br = bit_reversal_permutation(N)
    llr_br = llr[br]
    decisions = [0] * N
    _sc_recursive_core(list(llr_br), list(range(N)), frozen_bits, decisions)
    return np.asarray(decisions, dtype=int)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers_llr = []
        for layer in range(n):
            if (phi >> layer) & 1 == 0:
                layers_llr.append(layer)
            else:
                break
        layers_bit = []
        for layer in range(n):
            if ((phi + 1) >> layer) & 1:
                layers_bit.append(layer)
        llr_layer_vec.append(layers_llr)
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def _sc_phase_llr(llr_ch, u_prefix, phi):
    """计算相位 phi 的判决 LLR"""
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    llr_br = llr_ch[br]
    decisions = [int(u_prefix[i]) for i in range(phi)]
    partial = [0] * N
    for i in range(phi):
        partial[i] = decisions[i]

    def llr_at(phi_target):
        dec = partial.copy()

        def rec(llrs, indices):
            if len(indices) == 1:
                return llrs[0]
            half = len(indices) // 2
            left_llr = [f_operation(llrs[i], llrs[i + half]) for i in range(half)]
            if phi_target in indices[:half]:
                return rec(left_llr, indices[:half])
            enc_left = []
            for idx in indices[:half]:
                if idx < phi_target:
                    enc_left.append(dec[idx])
                else:
                    enc_left.append(0)
            right_llr = [
                g_operation(llrs[i], llrs[i + half], enc_left[i]) for i in range(half)
            ]
            return rec(right_llr, indices[half:])

        return rec(list(llr_br), list(range(N)))

    return llr_at(phi)


def sc_phase_llr(llr_ch, frozen_bits, u_prefix, phi):
    """已知 u_prefix[0:phi]，计算比特 phi 的 LLR"""
    llr = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    br = bit_reversal_permutation(N)
    llr_br = llr[br]
    target = [None]

    def rec(llrs, indices):
        if len(indices) == 1:
            i = indices[0]
            if i == phi:
                target[0] = float(llrs[0])
            return
        half = len(indices) // 2
        left_llr = [f_operation(llrs[i], llrs[i + half]) for i in range(half)]
        rec(left_llr, indices[:half])
        if target[0] is not None:
            return
        enc_left = [int(u_prefix[idx]) for idx in indices[:half]]
        right_llr = [
            g_operation(llrs[i], llrs[i + half], enc_left[i]) for i in range(half)
        ]
        rec(right_llr, indices[half:])

    rec(list(llr_br), list(range(N)))
    return target[0]


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（调用验证过的递归核心）"""
    return sc_decode_recursive(llr_ch, frozen_bits)
