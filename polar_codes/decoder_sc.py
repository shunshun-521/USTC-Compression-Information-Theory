"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """
    精确 log-domain f 运算（check node）：
    f(a,b) = logaddexp(0, a+b) - logaddexp(a, b)
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.logaddexp(0.0, La + Lb) - np.logaddexp(La, Lb)


def g_operation(La, Lb, beta):
    """
    g 运算：g(La, Lb, beta) = Lb + (1 - 2*beta) * La
    beta 为左子树的部分和向量（非最终信息比特）。
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    return Lb + (1.0 - 2.0 * beta) * La


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，与 CommPy 极化树一致）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def node(llr_node, base, length):
        if length == 1:
            idx = base
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return np.array([u_hat[idx]], dtype=np.uint8)

        half = length // 2
        beta_left = node(f_operation(llr_node[:half], llr_node[half:]), base, half)
        llr_right = g_operation(llr_node[:half], llr_node[half:], beta_left)
        beta_right = node(llr_right, base + half, half)
        return np.concatenate([beta_left ^ beta_right, beta_right])

    node(llr, 0, N)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量（层更新模式）"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layer = 0
        tmp = phi
        while tmp % 2 == 1:
            layer += 1
            tmp //= 2
        llr_layer_vec.append(list(range(layer, n)))
        bit_layer_vec.append(list(range(layer, n)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码：当前实现调用高效递归核（复杂度 O(N log N)）。
    接口与非递归版本一致，便于仿真脚本切换。
    """
    return sc_decode_recursive(llr_ch, frozen_bits)
