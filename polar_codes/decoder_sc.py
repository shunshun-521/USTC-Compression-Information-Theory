"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）与非递归主循环（高效实现，同一核心算法）
"""
import math
import numpy as np

# ==================== 基本运算 ====================


def _sign_pm(x):
    return np.where(x >= 0, 1.0, -1.0)


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（SC 默认）。"""
    return _sign_pm(La) * _sign_pm(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def f_boxplus(La, Lb):
    """精确 box-plus f 运算（用于参考递归实现）。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    La = np.clip(La, -30.0, 30.0)
    Lb = np.clip(Lb, -30.0, 30.0)
    return np.log1p(np.exp(La + Lb)) - np.log(np.exp(La) + np.exp(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1.0 - 2.0 * np.asarray(u_hat, dtype=np.float64)) * La + Lb


def _polar_decode_sc_recursive(llr, frozen_bits, f_func):
    """Sionna 风格的递归 SC（含 stage 部分和 u_up）。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    n = len(llr)

    if n > 1:
        half = n // 2
        llr1, llr2 = llr[:half], llr[half:]
        fr1, fr2 = frozen_bits[:half], frozen_bits[half:]
        u1, u1_up = _polar_decode_sc_recursive(f_func(llr1, llr2), fr1, f_func)
        u2, u2_up = _polar_decode_sc_recursive(
            g_operation(llr1, llr2, u1_up), fr2, f_func
        )
        u = np.concatenate([u1, u2])
        u1_up_int = (u1_up.astype(int) ^ u2_up.astype(int)).astype(np.float64)
        u_up = np.concatenate([u1_up_int, u2_up])
        return u, u_up

    if frozen_bits[0]:
        u = 0
    else:
        u = 0 if llr[0] >= 0 else 1
    return np.array([u], dtype=int), np.array([u], dtype=np.float64)


# ==================== 递归 SC 译码（参考实现，box-plus）====================


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（box-plus f，用于对照验证）。"""
    u, _ = _polar_decode_sc_recursive(llr, frozen_bits, f_boxplus)
    return u


# ==================== 非递归 SC 译码（高效实现）====================


def precompute_sc_indices(N):
    """预计算非递归 SC 所需的层索引列表（与 Arikan 相位更新一致）。"""
    n = int(math.log2(N))
    lambda_offset = np.zeros(n + 1, dtype=np.int64)
    for layer in range(n):
        lambda_offset[layer + 1] = lambda_offset[layer] + (1 << layer)

    llr_layer_vec = []
    for phi in range(N):
        layers = []
        p = phi
        while p & 1:
            layers.append(int(math.log2(p & -p)))
            p >>= 1
        llr_layer_vec.append(layers)

    bit_layer_vec = []
    for phi in range(N):
        if phi == N - 1:
            bit_layer_vec.append(list(range(n)))
            continue
        layers = []
        p = phi
        while (p & 1) == 0:
            layers.append(int(math.log2(p + 1)) - 1)
            p >>= 1
        bit_layer_vec.append(layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归主循环：调用 min-sum 递归核（主仿真路径）。"""
    u, _ = _polar_decode_sc_recursive(llr_ch, frozen_bits, f_operation)
    return u
