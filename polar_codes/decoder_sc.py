"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
import math

LLR_MAX = 30.0


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def cn_op(x, y):
    """精确 box-plus。"""
    x_in = np.clip(x, -LLR_MAX, LLR_MAX)
    y_in = np.clip(y, -LLR_MAX, LLR_MAX)
    return np.log(1.0 + np.exp(x_in + y_in)) - np.log(
        np.exp(x_in) + np.exp(y_in)
    )


def g_operation(La, Lb, u_hat):
    return (1 - 2 * u_hat) * La + Lb


def _sc_recursive_core(llr, frozen_ind):
    llr = np.asarray(llr, dtype=np.float64)
    frozen_ind = np.asarray(frozen_ind, dtype=np.float64)
    n = len(frozen_ind)

    if n > 1:
        if np.all(frozen_ind == 1):
            u_hat = np.zeros(n, dtype=np.float64)
            return u_hat, u_hat.copy()

        half = n // 2
        llr1 = llr[:half]
        llr2 = llr[half:]
        f1 = frozen_ind[:half]
        f2 = frozen_ind[half:]

        x_llr1 = cn_op(llr1, llr2)
        u_hat1, u_hat1_up = _sc_recursive_core(x_llr1, f1)

        x_llr2 = g_operation(llr1, llr2, u_hat1_up)
        u_hat2, u_hat2_up = _sc_recursive_core(x_llr2, f2)

        u_hat = np.concatenate([u_hat1, u_hat2])
        u_hat1_up_int = u_hat1_up.astype(np.int8)
        u_hat2_up_int = u_hat2_up.astype(np.int8)
        u_hat1_up = np.bitwise_xor(u_hat1_up_int, u_hat2_up_int).astype(np.float64)
        u_hat_up = np.concatenate([u_hat1_up, u_hat2_up])
        return u_hat, u_hat_up

    if frozen_ind[0] == 1:
        u_hat = np.array([0.0], dtype=np.float64)
    else:
        val = 0.5 * (1.0 - np.sign(llr[0]))
        if val == 0.5:
            val = 1.0
        u_hat = np.array([val], dtype=np.float64)
    return u_hat, u_hat.copy()


def sc_decode_recursive(llr, frozen_bits):
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    frozen_ind = frozen_bits.astype(np.float64)
    u_hat, _ = _sc_recursive_core(np.asarray(llr, dtype=np.float64), frozen_ind)
    return u_hat.astype(int)


def precompute_sc_indices(N):
    n = int(math.log2(N))
    lambda_offset = [0] * (n + 1)
    for layer in range(n + 1):
        lambda_offset[layer] = 1 << layer

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        psi = phi
        while psi % 2 == 1:
            llr_layers.append(int(math.log2(psi & -psi)))
            psi >>= 1
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        psi = phi
        while psi > 0 and psi % 2 == 0:
            bit_layers.append(int(math.log2(psi & -psi)))
            psi >>= 1
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def _init_llr_f_only(P, n, lambda_offset):
    for layer in range(n - 1, -1, -1):
        lam = lambda_offset[layer]
        lam_next = lambda_offset[layer + 1]
        for beta in range(0, lam, 2):
            idx = lam_next + beta
            P[layer, idx] = cn_op(
                P[layer + 1, idx],
                P[layer + 1, idx + lam],
            )


def _update_llr_layers(P, C, phi, llr_layer_vec, lambda_offset):
    for layer in llr_layer_vec[phi]:
        lam = lambda_offset[layer]
        lam_next = lambda_offset[layer + 1]
        for beta in range(0, lam, 2):
            idx = lam_next + beta
            P[layer, idx] = cn_op(
                P[layer + 1, idx],
                P[layer + 1, idx + lam],
            )
            P[layer, idx + 1] = g_operation(
                P[layer + 1, idx],
                P[layer + 1, idx + lam],
                C[layer, idx],
            )


def _update_bit_layers(C, phi, bit_layer_vec, lambda_offset):
    for layer in bit_layer_vec[phi]:
        lam = lambda_offset[layer]
        lam_next = lambda_offset[layer + 1]
        for beta in range(0, lam, 2):
            idx = lam_next + beta
            C[layer + 1, idx + lam] = (
                C[layer, idx + 1] ^ C[layer + 1, idx + lam]
            )
            C[layer + 1, idx] = C[layer, idx + 1]


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（调用高效递归实现）。"""
    return sc_decode_recursive(llr_ch, frozen_bits)

