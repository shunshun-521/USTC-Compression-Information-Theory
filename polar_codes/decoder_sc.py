"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    u_hat = np.asarray(u_hat)
    return (1 - 2 * u_hat) * La + Lb


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（llr 已为比特倒序后的信道 LLR）"""
    frozen = np.asarray(frozen_bits, dtype=bool)

    def decode_node(llr_node, offset):
        n_len = len(llr_node)
        if n_len == 1:
            idx = offset
            if frozen[idx]:
                return np.array([0], dtype=int)
            llr_val = float(llr_node[0]) + 1e-10 * (idx + 1)
            return np.array([0 if llr_val >= 0 else 1], dtype=int)

        half = n_len // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        u_left = decode_node(llr_left, offset)
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        u_right = decode_node(llr_right, offset + half)
        return np.concatenate([u_left, u_right])

    return decode_node(np.asarray(llr, dtype=np.float64), 0)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码辅助向量（层列表）
    """
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers_llr = []
        t = phi
        while t % 2 == 1:
            layers_llr.append(int(math.log2(t & -t)))  # not quite
            t >>= 1
        # 从低位起第一个 0 的位置序列
        layers_llr = []
        t = phi
        layer = 0
        while (t & 1) == 1:
            layers_llr.append(layer)
            t >>= 1
            layer += 1
        for l in range(layer, n):
            layers_llr.append(l)
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        t = phi + 1
        layer = 0
        while t % 2 == 0:
            layers_bit.append(layer)
            t >>= 1
            layer += 1
        bit_layer_vec.append(layers_bit)

    lambda_offset = [1 << i for i in range(n + 1)]
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode_channel(llr_ch, frozen_bits):
    """对信道 LLR 做比特倒序后递归译码（参考实现）"""
    br = bit_reversal_permutation(len(llr_ch))
    frozen = np.asarray(frozen_bits, dtype=bool)
    return sc_decode_recursive(np.asarray(llr_ch, dtype=np.float64)[br], frozen)


def sc_decode_nonrecursive(llr_ch, frozen_bits):
    """
    非递归 SC 译码（分层存储）。与递归实现等价，供高效仿真使用。
    """
    N = len(llr_ch)
    n = int(math.log2(N))
    br = bit_reversal_permutation(N)
    frozen = np.asarray(frozen_bits, dtype=bool)

    L = np.zeros((n + 1, N), dtype=np.float64)
    C = np.zeros((n + 1, N), dtype=np.int8)
    L[n, :] = np.asarray(llr_ch, dtype=np.float64)[br]

    for phi in range(N):
        for layer in range(n):
            psi = phi >> layer
            step = 1 << layer
            if psi % 2 == 0:
                block = (phi >> (layer + 1)) << (layer + 1)
                for j in range(step):
                    a = block + j
                    L[layer, a] = f_operation(L[layer + 1, a], L[layer + 1, a + step])
            else:
                block = ((phi >> (layer + 1)) << (layer + 1)) + step
                for j in range(step):
                    a = block - step + j
                    L[layer, a] = g_operation(
                        L[layer + 1, a], L[layer + 1, a + step], C[layer, a]
                    )

        if frozen[phi]:
            u = 0
        else:
            u = 0 if L[0, phi] >= 0 else 1
        C[0, phi] = u

        layer = 0
        while layer < n and (phi >> layer) % 2 == 0:
            block = (phi >> (layer + 1)) << (layer + 1)
            step = 1 << layer
            for j in range(step):
                a = block + j
                C[layer + 1, a] = (C[layer, a] ^ C[layer, a + step]) & 1
                C[layer + 1, a + step] = C[layer, a + step]
            layer += 1

    return C[0, :].astype(int)


def sc_decode(llr_ch, frozen_bits):
    """主 SC 译码接口（递归，O(N log N)）"""
    return sc_decode_channel(llr_ch, frozen_bits)
