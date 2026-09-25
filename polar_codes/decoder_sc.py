"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _butterfly_partial_sum(bits):
    """对子块比特做蝶形编码，得到 g 运算所需的部分和"""
    bits = np.asarray(bits, dtype=np.int8).copy()
    n = int(math.log2(len(bits)))
    step = 1
    for _ in range(n):
        for i in range(0, len(bits), 2 * step):
            left = bits[i:i + step]
            right = bits[i + step:i + 2 * step]
            bits[i:i + step] = (left ^ right) & 1
            bits[i + step:i + 2 * step] = right
        step <<= 1
    return bits


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    N = len(llr)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
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

        c_left = _butterfly_partial_sum(u_hat[bit_offset:bit_offset + half])
        llr_right = g_operation(llr_node[:half], llr_node[half:], c_left)
        decode_node(llr_right, bit_offset + half)

    decode_node(np.asarray(llr, dtype=np.float64), 0)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        llr_layers = []
        for layer in range(n):
            if ((phi >> layer) & 1) == 0:
                llr_layers.append(layer)
        llr_layer_vec.append(llr_layers)

        if phi % 2 == 0:
            bit_layers = []
        else:
            bit_layers = []
            psi = phi
            layer = 0
            while (psi & 1) == 1:
                bit_layers.append(layer)
                psi >>= 1
                layer += 1
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数（显式栈，复杂度 O(N log N)）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    u_hat = np.zeros(N, dtype=int)

    stack = [(llr_ch, 0, 0)]
    while stack:
        llr, offset, step = stack[-1]
        n = len(llr)
        if n == 1:
            idx = offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr[0] >= 0 else 1
            stack.pop()
            continue

        half = n // 2
        if step == 0:
            llr_left = f_operation(llr[:half], llr[half:])
            stack[-1] = (llr, offset, 1)
            stack.append((llr_left, offset, 0))
        elif step == 1:
            c_left = _butterfly_partial_sum(u_hat[offset:offset + half])
            llr_right = g_operation(llr[:half], llr[half:], c_left)
            stack[-1] = (llr, offset, 2)
            stack.append((llr_right, offset + half, 0))
        else:
            stack.pop()

    return u_hat


def compute_bit_llr(llr_ch, u_decoded, phi, N):
    """计算第 phi 个比特的 LLR（给定已译前缀）"""

    def recurse(llr_node, bit_start, bit_end):
        if bit_end - bit_start == 1:
            return llr_node[0]
        half = (bit_end - bit_start) // 2
        mid = bit_start + half
        if phi < mid:
            llr_left = f_operation(llr_node[:half], llr_node[half:])
            return recurse(llr_left, bit_start, mid)
        c_left = _butterfly_partial_sum(u_decoded[bit_start:mid])
        llr_right = g_operation(llr_node[:half], llr_node[half:], c_left)
        return recurse(llr_right, mid, bit_end)

    return recurse(np.asarray(llr_ch, dtype=np.float64), 0, N)


def channel_llr_to_decoder(llr_ch):
    """将信道 LLR 重排为译码器内部顺序（比特倒序）"""
    N = len(llr_ch)
    rev = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[rev]


def sc_decode_channel(llr_ch, frozen_bits):
    """对信道 LLR 做倒序后执行 SC 译码"""
    return sc_decode(channel_llr_to_decoder(llr_ch), frozen_bits)
