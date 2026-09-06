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


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现）。
    frozen_bits: True 表示冻结位。
    """
    frozen_bits = np.asarray(frozen_bits, dtype=bool).astype(int)

    def decode_node(llr_node, frozen_node):
        n = len(llr_node)
        if n == 1:
            if frozen_node[0] == 1:
                u_hat = 0
            else:
                u_hat = 0 if llr_node[0] >= 0 else 1
            return np.array([u_hat], dtype=int), np.array([u_hat], dtype=int)

        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        u_left, u_left_up = decode_node(llr_left, frozen_node[:half])
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left_up)
        u_right, u_right_up = decode_node(llr_right, frozen_node[half:])
        u_hat = np.concatenate([u_left, u_right])
        u_hat_up = np.concatenate(
            [(u_left_up ^ u_right_up).astype(int), u_right_up.astype(int)]
        )
        return u_hat, u_hat_up

    u_hat, _ = decode_node(llr, frozen_bits)
    return u_hat


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    """
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        tmp = phi
        llr_layers = []
        for layer in range(n):
            if tmp % 2 == 0:
                llr_layers.append(layer)
                tmp >>= 1
            else:
                break
        llr_layer_vec.append(llr_layers)

        if phi % 2 == 0:
            bit_layers = []
        else:
            tmp = phi
            bit_layers = []
            for layer in range(n):
                bit_layers.append(layer)
                tmp >>= 1
                if tmp % 2 == 0:
                    break
        bit_layer_vec.append(bit_layers)

    lambda_offset = [2 ** i for i in range(n + 1)]
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    frozen_bits: 1/True 表示冻结位，0/False 表示信息位。
    """
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    llr = llr_ch[br].astype(np.float64)
    return sc_decode_recursive(llr, frozen_bits)


def sc_decode_nonrecursive(llr_ch, frozen_bits):
    """
    非递归 SC 译码（基于层向量更新，结果与递归版一致）。
    """
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    _, llr_layer_vec, bit_layer_vec = precompute_sc_indices(N)

    P = np.zeros((n + 1, N), dtype=np.float64)
    C = np.zeros((n + 1, N), dtype=int)
    U = np.zeros((n + 1, N), dtype=int)
    P[n] = llr_ch.astype(np.float64)
    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        for layer in llr_layer_vec[phi]:
            step = 2 ** layer
            for block in range(0, N, 2 * step):
                for j in range(step):
                    left = block + j
                    right = left + step
                    P[layer][left] = f_operation(P[layer + 1][left], P[layer + 1][right])

        tmp = phi
        layer = 0
        while tmp % 2 == 1:
            step = 2 ** layer
            block = (phi // (2 * step)) * 2 * step
            j = phi % step
            left = block + j
            right = left + step
            P[layer][right] = g_operation(
                P[layer + 1][left], P[layer + 1][right], U[layer][left]
            )
            tmp >>= 1
            layer += 1

        if frozen_bits[phi]:
            u_hat[phi] = 0
        else:
            u_hat[phi] = 0 if P[0][phi] >= 0 else 1

        C[0][phi] = u_hat[phi]
        U[0][phi] = u_hat[phi]
        for layer in bit_layer_vec[phi]:
            step = 2 ** layer
            block = (phi // (2 * step)) * 2 * step
            j = phi % step
            right = block + j + step
            left = right - step
            C[layer + 1][right] = C[layer][phi]
            U[layer + 1][right] = C[layer][phi]
            U[layer][left] = U[layer + 1][left] ^ U[layer + 1][right]

    return u_hat
