"""
极化码 SC（串行抵消）译码器
"""
import numpy as np
import math


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat)
    return (1.0 - 2.0 * u_hat) * La + Lb


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, bit_offset, length):
        if length == 1:
            if frozen_bits[bit_offset]:
                u_hat[bit_offset] = 0
            else:
                u_hat[bit_offset] = 0 if llr_node[0] >= 0 else 1
            return

        half = length // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        decode_node(llr_left, bit_offset, half)

        u_left = u_hat[bit_offset:bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        decode_node(llr_right, bit_offset + half, half)

    decode_node(llr, 0, N)
    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """SC 译码主函数"""
    return sc_decode_recursive(llr_ch, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量（接口保留）"""
    n = int(math.log2(N))
    lambda_offset = [2 ** i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        phi_bin = format(phi, f"0{n}b")
        llr_layers = [l for l in range(n) if phi_bin[n - 1 - l] == "0"]
        llr_layer_vec.append(llr_layers)
        bit_layers = []
        for l in range(n):
            if l < n - 1:
                if phi_bin[n - 2 - l] == "1":
                    bit_layers.append(l)
            else:
                bit_layers.append(l)
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec
