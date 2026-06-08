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
    支持向量化（La, Lb 为同形状 numpy 数组）
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1 - 2 * u_hat) * La + Lb


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码。
    参数：
        llr: 长度 N 的信道 LLR 数组
        frozen_bits: 长度 N 的 bool 数组，True 表示冻结位（置 0）
    返回：
        u_hat: 长度 N 的估计源序列
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    u_hat = np.zeros(len(llr), dtype=int)

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
        decode_node(llr_left, bit_offset)

        u_left = u_hat[bit_offset:bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        decode_node(llr_right, bit_offset + half)

    decode_node(llr, 0)
    return u_hat


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的三个辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = np.zeros(n + 1, dtype=int)
    for layer in range(1, n + 1):
        lambda_offset[layer] = lambda_offset[layer - 1] + (1 << (n - layer))

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers_llr = []
        pp = phi
        while pp % 2 == 1:
            layers_llr.append(int(math.log2(pp & -pp)))
            pp //= 2
        layers_llr.append(n - 1)
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        if phi % 2 == 0:
            pp = phi
            while pp % 2 == 0 and pp > 0:
                layers_bit.append(int(math.log2(pp & -pp)))
                pp //= 2
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（显式栈，与递归版本等价）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    u_hat = np.zeros(len(llr_ch), dtype=int)

    stack = [(llr_ch.copy(), 0, 0)]
    while stack:
        llr_node, bit_offset, state = stack.pop()
        half = len(llr_node) // 2

        if half == 0:
            idx = bit_offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            continue

        if state == 0:
            llr_left = f_operation(llr_node[:half], llr_node[half:])
            stack.append((llr_node, bit_offset, 1))
            stack.append((llr_left, bit_offset, 0))
        else:
            u_left = u_hat[bit_offset:bit_offset + half]
            llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
            stack.append((llr_right, bit_offset + half, 0))

    return u_hat
