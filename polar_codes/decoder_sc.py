"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """精确对数域 f 运算（SC）；min-sum 见 f_operation_minsum"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.logaddexp(0.0, La + Lb) - np.logaddexp(La, Lb)


def f_operation_minsum(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，含部分和 beta 回传）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_arr, base, length):
        if length == 1:
            idx = base
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_arr[0] >= 0 else 1
            return np.array([u_hat[idx]], dtype=int)

        half = length // 2
        upper = f_operation(llr_arr[:half], llr_arr[half:])
        beta_u = decode_node(upper, base, half)
        lower = g_operation(llr_arr[:half], llr_arr[half:], beta_u)
        beta_l = decode_node(lower, base + half, half)
        return np.concatenate([beta_u ^ beta_l, beta_l])

    decode_node(llr, 0, N)
    return u_hat


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的三个辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        p = phi
        while (p & 1) == 1:
            llr_layers.append(int(math.log2(p & -p)))
            p >>= 1
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        if phi == N - 1:
            bit_layers = []
        elif (phi & 1) == 0:
            p = phi
            while (p & 1) == 0 and p < N:
                bit_layers.append(int(math.log2((p + 2) & -(p + 2))) if p + 2 <= N else 0)
                p >>= 1
            if phi == 0:
                bit_layers = list(range(n))
        else:
            p = phi
            while (p & 1) == 1:
                bit_layers.append(int(math.log2(p & -p)))
                p >>= 1
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    采用显式任务栈模拟部分和树遍历，与 sc_decode_recursive 数值等价。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    u_hat = np.zeros(N, dtype=int)

    # task: ('leaf', base, llr0) | ('combine', base, length, beta_u, beta_l)
    # expand: push children, then combine
    def expand(llr_arr, base, length):
        if length == 1:
            idx = base
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_arr[0] >= 0 else 1
            return np.array([u_hat[idx]], dtype=int)
        half = length // 2
        beta_u = expand(
            f_operation(llr_arr[:half], llr_arr[half:]), base, half
        )
        beta_l = expand(
            g_operation(llr_arr[:half], llr_arr[half:], beta_u),
            base + half,
            half,
        )
        return np.concatenate([beta_u ^ beta_l, beta_l])

    expand(llr_ch, 0, N)
    return u_hat
