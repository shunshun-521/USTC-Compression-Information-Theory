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
    return (1.0 - 2.0 * u_hat) * La + Lb


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（就地蝶形，与 sc_decode 等价）。
    """
    llr = np.asarray(llr, dtype=np.float64).copy()
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(offset, length, bit_pos):
        if length == 1:
            if frozen_bits[bit_pos]:
                u_hat[bit_pos] = 0
            else:
                u_hat[bit_pos] = 0 if llr[offset] >= 0 else 1
            return
        half = length // 2
        for i in range(half):
            llr[offset + i] = f_operation(llr[offset + i], llr[offset + half + i])
        decode_node(offset, half, bit_pos)
        for i in range(half):
            llr[offset + i] = g_operation(
                llr[offset + i], llr[offset + half + i], u_hat[bit_pos + i]
            )
        decode_node(offset + half, half, bit_pos + half)

    decode_node(0, N, 0)
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
        # 标准算法：找 phi 二进制末尾连续 1 之后的层直到 n-1
        layers_llr = []
        if phi == 0:
            layers_llr = list(range(n))
        else:
            tmp = phi
            layer = 0
            while tmp & 1:
                layer += 1
                tmp >>= 1
            layers_llr = list(range(layer, n))

        llr_layer_vec.append(layers_llr)

        # 比特回传层
        layers_bit = []
        if phi & 1:
            tmp = phi
            layer = 0
            while tmp & 1:
                layers_bit.append(layer)
                layer += 1
                tmp >>= 1
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（显式栈 + 就地蝶形，等价于递归 SC）。
    precompute_sc_indices 供分析与扩展；主路径使用栈调度。
    """
    llr = np.asarray(llr_ch, dtype=np.float64).copy()
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    # 帧: enter | after_left
    stack = [("enter", 0, N, 0)]
    while stack:
        frame, offset, length, bit_pos = stack.pop()
        if length == 1:
            if frozen_bits[bit_pos]:
                u_hat[bit_pos] = 0
            else:
                u_hat[bit_pos] = 0 if llr[offset] >= 0 else 1
            continue

        half = length // 2
        if frame == "enter":
            for i in range(half):
                llr[offset + i] = f_operation(
                    llr[offset + i], llr[offset + half + i]
                )
            stack.append(("after_left", offset, length, bit_pos))
            stack.append(("enter", offset, half, bit_pos))
        else:
            for i in range(half):
                llr[offset + i] = g_operation(
                    llr[offset + i],
                    llr[offset + half + i],
                    u_hat[bit_pos + i],
                )
            stack.append(("enter", offset + half, half, bit_pos + half))

    return u_hat


def sc_decode_mapped(llr_ch, frozen_bits):
    """对自然顺序信道 LLR 做倒序映射后 SC 译码。"""
    from encoder import map_channel_llr

    return sc_decode(map_channel_llr(llr_ch), frozen_bits)
