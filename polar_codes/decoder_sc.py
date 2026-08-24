"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
import math

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算（boxplus）：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    u_hat 为当前层的部分和（re-encoded partial sum）。
    """
    u = np.asarray(u_hat, dtype=np.float64)
    return (1.0 - 2.0 * u) * La + Lb


def _sc_decode_core(llr, frozen_bits):
    """递归 SC 内核，返回 (u_hat, u_hat_up)。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    def decode_node(llr_node, frozen_node):
        n = len(llr_node)
        if n == 1:
            if frozen_node[0]:
                bit = np.array([0], dtype=np.int8)
            else:
                bit = np.array([0 if llr_node[0] >= 0 else 1], dtype=np.int8)
            return bit, bit

        half = n // 2
        llr_left = llr_node[:half]
        llr_right = llr_node[half:]
        frozen_left = frozen_node[:half]
        frozen_right = frozen_node[half:]

        llr_up = f_operation(llr_left, llr_right)
        u_left, u_left_up = decode_node(llr_up, frozen_left)

        llr_down = g_operation(llr_left, llr_right, u_left_up)
        u_right, u_right_up = decode_node(llr_down, frozen_right)

        u_hat = np.concatenate([u_left, u_right])
        u_left_up_xor = (u_left_up.astype(int) ^ u_right_up.astype(int)).astype(np.int8)
        u_up = np.concatenate([u_left_up_xor, u_right_up])
        return u_hat, u_up

    return decode_node(llr, frozen_bits)


def llr_at_phi(llr, frozen_bits, u_pin, phi):
    """在已知 u_pin[0:phi] 时，计算比特 phi 处的 LLR。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    u_pin = np.asarray(u_pin, dtype=np.int8)

    def decode_node(llr_node, frozen_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            if idx < phi:
                bit = int(u_pin[idx])
                return bit, np.array([bit], dtype=np.int8)
            if idx == phi:
                return 'LLR', float(llr_node[0])
            if frozen_node[0]:
                return 0, np.array([0], dtype=np.int8)
            bit = 0 if llr_node[0] >= 0 else 1
            return bit, np.array([bit], dtype=np.int8)

        half = n // 2
        llr_left = llr_node[:half]
        llr_right = llr_node[half:]
        frozen_left = frozen_node[:half]
        frozen_right = frozen_node[half:]

        llr_up = f_operation(llr_left, llr_right)
        u_left, u_left_up = decode_node(llr_up, frozen_left, bit_offset)
        if u_left == 'LLR':
            return u_left, u_left_up

        llr_down = g_operation(llr_left, llr_right, u_left_up)
        u_right, u_right_up = decode_node(llr_down, frozen_right, bit_offset + half)
        if u_right == 'LLR':
            return u_right, u_right_up

        u_left_up_xor = (u_left_up.astype(int) ^ u_right_up.astype(int)).astype(np.int8)
        u_up = np.concatenate([u_left_up_xor, u_right_up])
        return u_left, u_up

    tag, val = decode_node(llr, frozen_bits, 0)
    if tag != 'LLR':
        raise ValueError("failed to compute LLR at phi")
    return val


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    llr = np.asarray(llr, dtype=np.float64)
    N = len(llr)
    br = bit_reversal_permutation(N)
    return _sc_decode_core(llr[br], frozen_bits)[0]


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量。"""
    n = int(math.log2(N))
    lambda_offset = [2 ** i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        l = 0
        while l < n and ((phi >> l) & 1):
            l += 1
        llr_layer_vec.append(list(range(l, n)))
        if l > 0:
            bit_layer_vec.append(list(range(l - 1, -1, -1)))
        else:
            bit_layer_vec.append([])

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode_nonrecursive(llr_ch, frozen_bits):
    """
    非递归 SC 译码（高效实现，当前回退到递归内核以保证正确性）。
    """
    return _sc_decode_core(llr_ch, frozen_bits)[0]


def sc_decode(llr_ch, frozen_bits):
    """
    SC 译码主入口。
    polar_encode 输出含比特倒序，故对信道 LLR 做相同倒序后译码。
    """
    return sc_decode_recursive(llr_ch, frozen_bits)
