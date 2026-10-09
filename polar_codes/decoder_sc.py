"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """f 运算：box-plus（比 min-sum 更适合高 SNR / 长码长）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    ta = np.tanh(La / 2.0)
    tb = np.tanh(Lb / 2.0)
    prod = np.clip(ta * tb, -1.0 + 1e-12, 1.0 - 1e-12)
    return 2.0 * np.arctanh(prod)


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

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

        u_left = u_hat[bit_offset : bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        decode_node(llr_right, bit_offset + half)

    decode_node(llr, 0)
    return u_hat


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码辅助向量。
    返回 lambda_offset, llr_layer_vec, bit_layer_vec
    """
    n = int(math.log2(N))
    if 2 ** n != N:
        raise ValueError("N must be a power of 2")

    lambda_offset = [2 ** layer for layer in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        p = phi
        layer = 0
        while p & 1:
            layer += 1
            p >>= 1
        for l in range(layer, n):
            llr_layers.append(l)

        bit_layers = []
        p = phi >> 1
        layer = 0
        while p & 1:
            layer += 1
            p >>= 1
        for l in range(layer, n):
            bit_layers.append(l)

        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（显式栈模拟递归 DFS，与 sc_decode_recursive 等价）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    u_hat = np.zeros(len(llr_ch), dtype=int)

    stack = [("enter", llr_ch, 0)]
    while stack:
        tag, *payload = stack.pop()
        if tag == "enter":
            llr_node, bit_offset = payload
            if len(llr_node) == 1:
                idx = bit_offset
                if frozen_bits[idx]:
                    u_hat[idx] = 0
                else:
                    u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            else:
                half = len(llr_node) // 2
                llr_left = f_operation(llr_node[:half], llr_node[half:])
                stack.append(("after_left", llr_node, bit_offset, half))
                stack.append(("enter", llr_left, bit_offset))
        else:  # after_left
            llr_node, bit_offset, half = payload
            u_left = u_hat[bit_offset : bit_offset + half]
            llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
            stack.append(("enter", llr_right, bit_offset + half))

    return u_hat
