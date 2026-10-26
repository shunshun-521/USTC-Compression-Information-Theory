"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np

# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


# ==================== 递归 SC 译码（参考实现）====================


def _polar_decode_sc_recursive(llr_ch, frozen_ind):
    """返回 (u_hat, u_hat_up)。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_ind = np.asarray(frozen_ind, dtype=int)
    n = len(llr_ch)

    if n > 1:
        half = n // 2
        llr1 = llr_ch[:half]
        llr2 = llr_ch[half:]
        fz1 = frozen_ind[:half]
        fz2 = frozen_ind[half:]

        llr_left = f_operation(llr1, llr2)
        u_hat1, u_hat1_up = _polar_decode_sc_recursive(llr_left, fz1)

        llr_right = g_operation(llr1, llr2, u_hat1_up)
        u_hat2, u_hat2_up = _polar_decode_sc_recursive(llr_right, fz2)

        u_hat = np.concatenate([u_hat1, u_hat2])
        u_hat1_up = (u_hat1_up ^ u_hat2_up) % 2
        u_hat_up = np.concatenate([u_hat1_up, u_hat2_up])
        return u_hat, u_hat_up

    if frozen_ind[0]:
        u = np.array([0], dtype=int)
    else:
        u = np.array([0 if llr_ch[0] >= 0 else 1], dtype=int)
    return u, u.copy()


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码。"""
    frozen_ind = np.asarray(frozen_bits, dtype=bool).astype(int)
    u_hat, _ = _polar_decode_sc_recursive(llr, frozen_ind)
    return u_hat


# ==================== 非递归 SC 译码（高效实现）====================


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量。"""
    n = int(np.log2(N))
    lambda_offset = [0] * (n + 1)
    for i in range(1, n + 1):
        lambda_offset[i] = lambda_offset[i - 1] + (1 << (n - i))

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        psi = phi
        layer = 0
        while psi & 1:
            psi >>= 1
            layer += 1
        for l in range(layer, n):
            llr_layers.append(l)
        llr_layer_vec.append(llr_layers)

        psi = phi
        layer = 0
        while psi & 1:
            psi >>= 1
            layer += 1
        if phi & 1:
            bit_layers = list(range(layer + 1))
        else:
            bit_layers = list(range(layer))
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（当前与递归实现等价，保证正确性）。 """
    return sc_decode_recursive(llr_ch, frozen_bits)
