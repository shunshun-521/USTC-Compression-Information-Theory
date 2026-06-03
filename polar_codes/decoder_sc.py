"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """f 运算：大 LLR 用 min-sum，否则用精确 box-plus。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    sa, sb = np.sign(La), np.sign(Lb)
    aa, ab = np.abs(La), np.abs(Lb)
    ms = sa * sb * np.minimum(aa, ab)
    with np.errstate(over="ignore", invalid="ignore"):
        t = np.tanh(aa / 2.0) * np.tanh(ab / 2.0)
        t = np.clip(t, -1.0 + 1e-15, 1.0 - 1e-15)
        exact = 2.0 * np.arctanh(t)
        exact = sa * sb * exact
    use_ms = (aa > 30.0) | (ab > 30.0)
    return np.where(use_ms, ms, exact)


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    return (1.0 - 2.0 * u_hat) * La + Lb


# ==================== 递归 SC 译码（参考实现）====================


def _sc_decode_core(llr, frozen_bits, offset, n):
    """LLR 长度 2^n，译码比特区间 [offset, offset + 2^n)。"""
    if n == 0:
        idx = offset
        if frozen_bits[idx]:
            return np.array([0], dtype=int)
        return np.array([0 if llr[0] >= 0 else 1], dtype=int)

    half = 1 << (n - 1)
    llr_left = np.array(
        [f_operation(llr[2 * i], llr[2 * i + 1]) for i in range(half)]
    )
    u_left = _sc_decode_core(llr_left, frozen_bits, offset, n - 1)
    llr_right = np.array(
        [
            g_operation(llr[2 * i], llr[2 * i + 1], u_left[i])
            for i in range(half)
        ]
    )
    u_right = _sc_decode_core(llr_right, frozen_bits, offset + half, n - 1)
    return np.concatenate([u_left, u_right])


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    n = int(math.log2(len(llr)))
    return _sc_decode_core(llr, frozen_bits, 0, n)


# ==================== 非递归 SC 译码（高效实现）====================


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量。"""
    n = int(np.log2(N))
    lambda_offset = [1 << layer for layer in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        llr_layers = []
        p = phi
        layer = 0
        while layer < n:
            if p % 2 == 0:
                llr_layers.append(layer)
                p //= 2
                layer += 1
            else:
                break
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        if phi % 2 == 1:
            p = (phi - 1) // 2
            layer = 0
            while layer < n:
                bit_layers.append(layer)
                if p % 2 == 0:
                    break
                p //= 2
                layer += 1
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码。
    λ[l] 长度为 2^l（l=0 为根，l=n 为信道），β[l] 为部分和。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    lam = [np.zeros(1 << layer, dtype=np.float64) for layer in range(n + 1)]
    beta = [np.zeros(1 << layer, dtype=np.int8) for layer in range(n + 1)]
    lam[n][:] = llr_ch
    u_hat = np.zeros(N, dtype=int)

    def calc_lambda(layer, phase):
        if layer == 0:
            return
        if phase % 2 == 0:
            calc_lambda(layer - 1, phase // 2)
        for j in range(1 << (layer - 1)):
            if phase % 2 == 0:
                lam[layer - 1][j] = f_operation(lam[layer][2 * j], lam[layer][2 * j + 1])
            else:
                lam[layer - 1][j] = g_operation(
                    lam[layer][2 * j], lam[layer][2 * j + 1], beta[layer - 1][j]
                )

    def update_beta(layer, phase, bit):
        if layer == n:
            return
        for j in range(1 << layer):
            if phase % 2 == 0:
                beta[layer + 1][2 * j] = beta[layer][j]
                beta[layer + 1][2 * j + 1] = beta[layer][j] ^ bit
            else:
                beta[layer + 1][2 * j + 1] = beta[layer][j]
        if phase % 2 == 1:
            update_beta(layer + 1, phase // 2, bit)

    for phi in range(N):
        calc_lambda(n, phi)
        if frozen_bits[phi]:
            u_hat[phi] = 0
        else:
            u_hat[phi] = 0 if lam[0][0] >= 0 else 1
        beta[0][0] = u_hat[phi]
        update_beta(0, phi, u_hat[phi])

    return u_hat


# 默认使用递归 SC（已充分验证）；sc_decode_fast 为层矩阵非递归实现
sc_decode_fast = sc_decode
sc_decode = sc_decode_recursive
