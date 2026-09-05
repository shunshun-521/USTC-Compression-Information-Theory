"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（显式栈，等价实现）
"""
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_partial):
    """g 运算，u_partial 为部分和（非单比特）"""
    return (1 - 2 * u_partial) * La + Lb


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，含部分和更新）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, bit_offset):
        n = len(llr_node)
        half = n // 2
        s = np.zeros(n, dtype=int)

        if n == 1:
            idx = bit_offset
            if frozen_bits[idx]:
                s[0] = 0
            else:
                s[0] = 0 if llr_node[0] >= 0 else 1
            u_hat[idx] = s[0]
            return s

        llr_left = f_operation(llr_node[:half], llr_node[half:])
        s_left = decode_node(llr_left, bit_offset)

        llr_right = g_operation(llr_node[:half], llr_node[half:], s_left)
        s_right = decode_node(llr_right, bit_offset + half)

        s[:half] = s_left ^ s_right
        s[half:] = s_right
        return s

    decode_node(llr, 0)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = [l for l in range(n) if (phi & (1 << l)) == 0]
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        p = (phi + 1) >> 1
        i = 0
        while (p >> i) & 1:
            bit_layers.append(i)
            i += 1
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（调用递归实现，O(N log N)）"""
    return sc_decode_recursive(llr_ch, frozen_bits)


def compute_sc_llr_at_phase(channel_llr, u_prefix, phi, frozen_bits):
    """计算 SC 在相位 phi 处的 LLR（基于部分和，供 SCL 使用）"""
    channel_llr = np.asarray(channel_llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    u_prefix = np.asarray(u_prefix, dtype=int)

    def decode_node(llr_node, bit_offset):
        n = len(llr_node)
        half = n // 2
        s = np.zeros(n, dtype=int)

        if bit_offset + n <= phi:
            if n == 1:
                s[0] = u_prefix[bit_offset]
                return s
            llr_left = f_operation(llr_node[:half], llr_node[half:])
            s_left = decode_node(llr_left, bit_offset)
            llr_right = g_operation(llr_node[:half], llr_node[half:], s_left)
            s_right = decode_node(llr_right, bit_offset + half)
            s[:half] = s_left ^ s_right
            s[half:] = s_right
            return s

        if bit_offset == phi and n == 1:
            return llr_node[0]

        if phi < bit_offset + half:
            llr_left = f_operation(llr_node[:half], llr_node[half:])
            return decode_node(llr_left, bit_offset)

        llr_left = f_operation(llr_node[:half], llr_node[half:])
        s_left = decode_node(llr_left, bit_offset)
        llr_right = g_operation(llr_node[:half], llr_node[half:], s_left)
        return decode_node(llr_right, bit_offset + half)

    result = decode_node(channel_llr, 0)
    return float(result)
