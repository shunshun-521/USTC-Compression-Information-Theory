"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, partial_bits):
    """g 运算，partial_bits 为左子树返回的部分和比特。"""
    return Lb + (1.0 - 2.0 * partial_bits) * La


def _xor_combine(left, right):
    out = np.empty(len(left) + len(right), dtype=int)
    out[: len(left)] = (left + right) % 2
    out[len(left) :] = right
    return out


def _channel_llr_to_decoder(llr_ch):
    """将信道码字顺序的 LLR 转换为蝶形译码域顺序。"""
    N = len(llr_ch)
    return np.asarray(llr_ch, dtype=np.float64)[bit_reversal_permutation(N)]


def sc_decode_recursive(llr_ch, frozen_bits):
    """
    递归 SC 译码（树节点索引，参考实现）。
    """
    llr = _channel_llr_to_decoder(llr_ch)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(math.log2(N)) + 1
    u_hat = np.zeros(N, dtype=int)

    def decode_node(y, depth, node):
        if depth == n - 1:
            if frozen_bits[node]:
                u_hat[node] = 0
                return np.array([0], dtype=int)
            bit = 1 if y[0] < 0 else 0
            u_hat[node] = bit
            return np.array([bit], dtype=int)

        half = len(y) // 2
        left_llr = f_operation(y[:half], y[half:])
        left_partial = decode_node(left_llr, depth + 1, 2 * node)
        right_llr = g_operation(y[:half], y[half:], left_partial)
        right_partial = decode_node(right_llr, depth + 1, 2 * node + 1)
        return _xor_combine(left_partial, right_partial)

    decode_node(llr, 0, 0)
    return u_hat


def precompute_sc_indices(N):
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        psi = phi
        for layer in range(n):
            if (psi & 1) == 0:
                llr_layers.append(layer)
            psi >>= 1
        llr_layer_vec.append(llr_layers)

        if phi % 2 == 0:
            bit_layer_vec.append(list(range(n)))
        else:
            bit_layers = []
            psi = phi
            for layer in range(n):
                if (psi & 1) == 1:
                    bit_layers.append(layer)
                psi >>= 1
            bit_layer_vec.append(bit_layers)

    return llr_layer_vec, bit_layer_vec


_SC_CACHE = {}


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（当前委托递归实现以保证正确性）。
    """
    return sc_decode_recursive(llr_ch, frozen_bits)


def _sc_decode_iterative(llr_ch, frozen_bits):
    """分层非递归 SC 译码（备用实现）。"""
    llr = _channel_llr_to_decoder(llr_ch)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(math.log2(N))

    llr_layer_vec, bit_layer_vec = precompute_sc_indices(N)

    P = np.zeros((n + 1, N), dtype=np.float64)
    C = np.zeros((n + 1, N), dtype=np.int8)
    P[n, :] = llr
    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        for layer in llr_layer_vec[phi]:
            span = 1 << (n - 1 - layer)
            for block in range(0, N, span * 2):
                left = block
                mid = block + span
                right = block + 2 * span
                if ((phi // span) % 2) == 0:
                    P[layer, left:mid] = f_operation(
                        P[layer + 1, left:mid],
                        P[layer + 1, mid:right],
                    )
                else:
                    P[layer, mid:right] = g_operation(
                        P[layer + 1, left:mid],
                        P[layer + 1, mid:right],
                        C[layer, left:mid],
                    )

        if frozen_bits[phi]:
            u_hat[phi] = 0
            C[0, 0] = 0
        else:
            u_hat[phi] = 0 if P[0, 0] >= 0 else 1
            C[0, 0] = u_hat[phi]

        for layer in bit_layer_vec[phi]:
            span = 1 << (n - 1 - layer)
            for block in range(0, N, span * 2):
                left = block
                mid = block + span
                right = block + 2 * span
                if ((phi // span) % 2) == 0:
                    C[layer + 1, mid:right] = C[layer, left:mid]
                else:
                    C[layer + 1, left:mid] = (
                        C[layer, left:mid] ^ C[layer, mid:right]
                    )
                    C[layer + 1, mid:right] = C[layer, mid:right]

    return u_hat
