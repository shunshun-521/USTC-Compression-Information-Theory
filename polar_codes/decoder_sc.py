"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """
    精确 log-domain f 运算（boxplus）：
    f(a,b) = ln((1+e^(a+b))/(e^a+e^b))
    """
    return np.logaddexp(0.0, La + Lb) - np.logaddexp(La, Lb)


def f_operation_minsum(La, Lb):
    """min-sum 近似的 f 运算（供 BP 等使用）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, beta):
    """
    g 运算：g(La, Lb, beta) = Lb + (1 - 2*beta) * La
    beta 为左子树的部分和（partial sum），非原始消息比特。
    """
    beta = np.asarray(beta, dtype=np.float64)
    return Lb + (1.0 - 2.0 * beta) * La


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现，使用精确 f 与部分和 beta）。
    """
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    def decode_node(llr_node, base, length):
        if length == 1:
            idx = base
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return np.array([u_hat[idx]], dtype=np.uint8)

        half = length // 2
        llr_upper = f_operation(llr_node[:half], llr_node[half:])
        beta_upper = decode_node(llr_upper, base, half)
        llr_lower = g_operation(llr_node[:half], llr_node[half:], beta_upper)
        beta_lower = decode_node(llr_lower, base + half, half)
        return np.concatenate([beta_upper ^ beta_lower, beta_lower])

    decode_node(llr, 0, N)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers_llr = []
        bits = format(phi, f"0{n}b")
        for s in range(n):
            if bits[n - 1 - s] == "0":
                layers_llr.append(s)
        llr_layer_vec.append(layers_llr)

        layers_bit = list(range(n)) if phi % 2 == 1 else [
            s for s in range(n) if bits[n - 1 - s] == "1"
        ]
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    SC 译码主函数。
    对信道 LLR 做比特倒序以匹配编码器输出顺序，再执行递归 SC 译码。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    rev = bit_reversal_permutation(len(llr_ch))
    return sc_decode_recursive(llr_ch[rev], frozen_bits)
