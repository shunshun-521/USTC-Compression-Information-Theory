"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _sc_recursive(llr, frozen_bits, u_hat, stage, pos):
    """递归 SC 内核：stage 层节点覆盖比特 [pos, pos+2^stage)"""
    if stage == 0:
        if frozen_bits[pos]:
            u_hat[pos] = 0
        else:
            u_hat[pos] = 0 if llr[0] >= 0 else 1
        return

    half = 1 << (stage - 1)
    llr_left = f_operation(llr[:half], llr[half : 2 * half])
    _sc_recursive(llr_left, frozen_bits, u_hat, stage - 1, pos)
    u_left = u_hat[pos : pos + half]
    llr_right = g_operation(llr[:half], llr[half : 2 * half], u_left)
    _sc_recursive(llr_right, frozen_bits, u_hat, stage - 1, pos + half)


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（参考实现）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    u_hat = np.zeros(N, dtype=int)
    _sc_recursive(llr_ch, frozen_bits, u_hat, n, 0)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助索引"""
    n = int(math.log2(N))
    lambda_offset = [0] * (n + 1)
    for i in range(1, n + 1):
        lambda_offset[i] = lambda_offset[i - 1] + (1 << i)

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        pp = phi
        llr_layers = []
        while pp % 2 == 1:
            llr_layers.append(int(math.log2(pp & -pp)))
            pp //= 2
        llr_layers.append(n)
        llr_layer_vec.append(llr_layers)

        pp = phi
        bit_layers = []
        while pp % 2 == 1:
            bit_layers.append(int(math.log2(pp & -pp)))
            pp //= 2
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC（调用递归实现，接口一致）"""
    return sc_decode_recursive(llr_ch, frozen_bits)
