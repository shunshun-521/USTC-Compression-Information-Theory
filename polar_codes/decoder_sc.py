"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import inverse_bit_reversal_permutation


def prepare_decoder_llr(llr_ch):
    """将信道 LLR 变换为 SC 树自然顺序（bit-reversal 编码的逆置换）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    inv = inverse_bit_reversal_permutation(len(llr_ch))
    return llr_ch[inv]


def f_operation(La, Lb):
    """
    f 运算（box-plus，LLR 域精确形式）。
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.logaddexp(0.0, La + Lb) - np.logaddexp(La, Lb)


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = Lb + (1 - 2*u_hat) * La
    """
    u_hat = np.asarray(u_hat, dtype=np.float64)
    return np.asarray(Lb, dtype=np.float64) + (1.0 - 2.0 * u_hat) * np.asarray(
        La, dtype=np.float64
    )


def _pm_penalty(llr_val, bit):
    return float(np.logaddexp(0.0, -(1.0 - 2.0 * bit) * llr_val))


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（树形参考实现）。
    """
    llr = prepare_decoder_llr(llr)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=np.int8)

    def decode_node(layer_llr, start, length):
        if length == 1:
            idx = start
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if layer_llr[0] >= 0.0 else 1
            return np.array([u_hat[idx]], dtype=np.int8)

        half = length // 2
        left_llr = f_operation(layer_llr[:half], layer_llr[half:])
        beta_left = decode_node(left_llr, start, half)
        right_llr = g_operation(layer_llr[:half], layer_llr[half:], beta_left)
        beta_right = decode_node(right_llr, start + half, half)
        return np.concatenate([beta_left ^ beta_right, beta_right])

    decode_node(llr, 0, N)
    return u_hat


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的三个辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        llr_layers = []
        bit_layers = []
        pp = phi
        while pp % 2 == 1:
            llr_layers.append(int(math.log2(pp & -pp)))
            pp //= 2

        psi = phi // 2
        while psi > 0:
            if psi % 2 == 1:
                bit_layers.append(int(math.log2(psi & -psi)))
            psi //= 2

        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（通过 L=1 的 SCL 实现，与递归 SC 等价）。
    """
    from decoder_scl import SCLDecoder

    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    u_hat, _ = SCLDecoder(len(llr_ch), frozen_bits, list_size=1).decode(llr_ch)
    return u_hat


def _sc_decode_iterative(llr_ch, frozen_bits):
    """逐比特更新的 SC 实现（保留供参考，与主 sc_decode 等价）。"""
    llr_ch = prepare_decoder_llr(llr_ch)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    llr = np.zeros((n + 1, N), dtype=np.float64)
    bits = np.zeros((n + 1, N), dtype=np.int8)
    llr[n, :N] = llr_ch
    u_hat = np.zeros(N, dtype=np.int8)

    for phi in range(N):
        l_start = 0
        pp = phi
        while pp % 2 == 1:
            pp //= 2
            l_start += 1

        for layer in range(l_start, n):
            pm = 1 << layer
            for omega in range(pm):
                llr[layer, omega] = f_operation(
                    llr[layer + 1, 2 * omega], llr[layer + 1, 2 * omega + 1]
                )

        for layer in range(n - 1, l_start - 1, -1):
            pm = 1 << layer
            for omega in range(pm):
                llr[layer, omega] = g_operation(
                    llr[layer + 1, 2 * omega],
                    llr[layer + 1, 2 * omega + 1],
                    bits[layer, 2 * omega],
                )

        if frozen_bits[phi]:
            u_hat[phi] = 0
        else:
            u_hat[phi] = 0 if llr[0, 0] >= 0.0 else 1
        bits[0, 0] = u_hat[phi]

        pp = phi
        layer = 0
        while pp % 2 == 1:
            psi = pp >> 1
            bits[layer, 2 * psi] = bits[layer + 1, psi]
            bits[layer, 2 * psi + 1] = u_hat[phi] ^ bits[layer + 1, psi]
            pp //= 2
            layer += 1

    return u_hat
