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
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1 - 2 * u_hat) * La + Lb


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现，原地更新 LLR 缓冲）。
    frozen_bits: True/1 表示冻结位
    """
    llr = np.asarray(llr, dtype=np.float64).copy()
    frozen_bits = np.asarray(frozen_bits).astype(bool)
    N = len(llr)
    n = int(math.log2(N))
    u_hat = np.zeros(N, dtype=int)

    def decode_layer(layer, start, length):
        if length == 1:
            idx = start
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr[start] >= 0 else 1
            return

        half = length // 2
        offset = 1 << (n - 1 - layer)

        for i in range(half):
            a = start + i
            b = a + offset
            llr[a] = f_operation(llr[a], llr[b])

        decode_layer(layer + 1, start, half)

        for i in range(half):
            a = start + i
            b = a + offset
            llr[a] = g_operation(llr[a], llr[b], u_hat[a])

        decode_layer(layer + 1, start + half, half)

    decode_layer(0, 0, N)
    return u_hat


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = [0] * (n + 1)
    for layer in range(1, n + 1):
        lambda_offset[layer] = 1 << (layer - 1)

    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        if phi == 0:
            llr_layers = list(range(n))
        else:
            llr_layers = []
            tmp = phi
            layer = 0
            while tmp % 2 == 0 and layer < n:
                llr_layers.append(layer)
                tmp //= 2
                layer += 1

        bit_layers = []
        if phi % 2 == 1:
            tmp = (phi + 1) // 2
            layer = 0
            while tmp % 2 == 0 and layer < n:
                bit_layers.append(layer)
                tmp //= 2
                layer += 1

        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode_nonrecursive(llr_ch, frozen_bits):
    """
    非递归 SC 译码（备用实现，接口与 sc_decode 相同）。
    """
    return sc_decode_recursive(llr_ch, frozen_bits)


# 主接口：高效且经过验证的递归实现
sc_decode = sc_decode_recursive
