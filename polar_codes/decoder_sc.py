"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现）。
    """
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    def decode_node(llr_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return

        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        for i in range(half):
            decode_node(llr_left[i : i + 1], bit_offset + i)
        u_left = u_hat[bit_offset : bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        for i in range(half):
            decode_node(llr_right[i : i + 1], bit_offset + half + i)

    decode_node(np.asarray(llr, dtype=np.float64), 0)
    return u_hat


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers_llr = []
        if phi == 0:
            layers_llr = list(range(n - 1, -1, -1))
        else:
            psi = phi
            layer = 0
            while psi % 2 == 1 and layer < n:
                layers_llr.append(layer)
                psi >>= 1
                layer += 1
            if layer < n:
                layers_llr.append(layer)
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        if phi % 2 == 0:
            psi = phi // 2
            layer = 0
            while psi % 2 == 1 and layer < n:
                layers_bit.append(layer)
                psi >>= 1
                layer += 1
            if layer < n:
                layers_bit.append(layer)
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


_SC_CACHE = {}


def _get_sc_tables(N):
    if N not in _SC_CACHE:
        _SC_CACHE[N] = precompute_sc_indices(N)
    return _SC_CACHE[N]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（高效实现）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(math.log2(N))
    lambda_offset, llr_layer_vec, bit_layer_vec = _get_sc_tables(N)

    m = n + 1
    P = np.zeros((m, 2 * N), dtype=np.float64)
    C = np.zeros((m, 2 * N), dtype=np.int32)
    P[m - 1, :N] = llr_ch

    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        for layer in llr_layer_vec[phi]:
            psi = phi >> layer
            delta = lambda_offset[layer]
            for omega in range(delta):
                idx = omega + psi * 2 * delta
                P[layer, idx] = f_operation(
                    P[layer + 1, idx], P[layer + 1, idx + delta]
                )
                P[layer, idx + delta] = g_operation(
                    P[layer + 1, idx],
                    P[layer + 1, idx + delta],
                    C[layer, idx],
                )

        if frozen_bits[phi]:
            u_hat[phi] = 0
            C[0, phi * 2] = 0
        else:
            u_hat[phi] = 0 if P[0, phi * 2] >= 0 else 1
            C[0, phi * 2] = u_hat[phi]

        for layer in bit_layer_vec[phi]:
            psi = phi >> layer
            delta = lambda_offset[layer]
            omega = psi % 2
            idx = (psi // 2) * 2 * delta + omega * delta
            C[layer + 1, 2 * idx] = C[layer, idx]
            C[layer + 1, 2 * idx + delta] = C[layer, idx] ^ C[layer, idx + delta]

    return u_hat
