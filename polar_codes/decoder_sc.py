"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

LLR_MAX = 30.0


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（CN / boxplus）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def f_operation_exact(La, Lb):
    """精确 boxplus（用于校验）"""
    x = np.clip(La, -LLR_MAX, LLR_MAX)
    y = np.clip(Lb, -LLR_MAX, LLR_MAX)
    return np.log1p(np.exp(x + y)) - np.log(np.exp(x) + np.exp(y))


def g_operation(La, Lb, u_partial):
    """VN / g 运算，u_partial 为当前层的部分和（非源比特）"""
    return (1 - 2 * u_partial) * La + Lb


def _sc_decode_recursive_core_forced(llr_ch, frozen_ind, u_prefix, phi, offset, f_op):
    """递归 SC，索引 < phi 的比特强制为 u_prefix 中的值。"""
    n = len(llr_ch)
    frozen_ind = np.asarray(frozen_ind, dtype=bool)
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    u_prefix = np.asarray(u_prefix, dtype=int)

    if n == 1:
        idx = offset
        if frozen_ind[0]:
            u = 0.0
        elif idx < phi:
            u = float(u_prefix[idx])
        else:
            u = 0.0 if llr_ch[0] >= 0 else 1.0
        return np.array([u]), np.array([u])

    half = n // 2
    llr1 = llr_ch[:half]
    llr2 = llr_ch[half:]
    f1 = frozen_ind[:half]
    f2 = frozen_ind[half:]

    u1, u1_up = _sc_decode_recursive_core_forced(
        f_op(llr1, llr2), f1, u_prefix, phi, offset, f_op
    )
    u2, u2_up = _sc_decode_recursive_core_forced(
        g_operation(llr1, llr2, u1_up), f2, u_prefix, phi, offset + half, f_op
    )

    u_hat = np.concatenate([u1, u2])
    u1_up_i = (u1_up.astype(np.int8) ^ u2_up.astype(np.int8)).astype(np.float64)
    u_hat_up = np.concatenate([u1_up_i, u2_up])
    return u_hat, u_hat_up


def _sc_decode_recursive_core(llr_ch, frozen_ind, f_op):
    """Sionna/Arikan 递归 SC，返回 (u_hat, u_hat_up)。"""
    n = len(llr_ch)
    frozen_ind = np.asarray(frozen_ind, dtype=bool)
    llr_ch = np.asarray(llr_ch, dtype=np.float64)

    if n == 1:
        if frozen_ind[0]:
            u = np.array([0.0])
        else:
            u = np.array([0.0 if llr_ch[0] >= 0 else 1.0])
        return u, u.copy()

    half = n // 2
    llr1 = llr_ch[:half]
    llr2 = llr_ch[half:]
    f1 = frozen_ind[:half]
    f2 = frozen_ind[half:]

    u1, u1_up = _sc_decode_recursive_core(f_op(llr1, llr2), f1, f_op)
    u2, u2_up = _sc_decode_recursive_core(g_operation(llr1, llr2, u1_up), f2, f_op)

    u_hat = np.concatenate([u1, u2])
    u1_up_i = (u1_up.astype(np.int8) ^ u2_up.astype(np.int8)).astype(np.float64)
    u_hat_up = np.concatenate([u1_up_i, u2_up])
    return u_hat, u_hat_up


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，精确 boxplus）"""
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    u_hat, _ = _sc_decode_recursive_core(llr, frozen_bits, f_operation_exact)
    return u_hat.astype(int)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量（供 SCL 参考）"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        tmp = phi
        layers = []
        while (tmp & 1) == 1 and len(layers) < n:
            layers.append(len(layers))
            tmp >>= 1
        llr_layer_vec.append(layers)
        bit_layer_vec.append(list(range(len(layers))))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """SC 译码主函数（O(N log N)，min-sum）"""
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    u_hat, _ = _sc_decode_recursive_core(llr_ch, frozen_bits, f_operation)
    return u_hat.astype(int)


def llr_at_phase(llr_ch, frozen_bits, u_prefix, phi, f_op=f_operation):
    """计算第 phi 个比特的 LLR（路径相关，用于 SCL）"""

    def rec(llr, frozen_seg, offset):
        n = len(llr)
        if n == 1:
            return float(llr[0])
        half = n // 2
        if phi < offset + half:
            return rec(f_op(llr[:half], llr[half:]), frozen_seg[:half], offset)
        _, u1_up = _sc_decode_recursive_core_forced(
            f_op(llr[:half], llr[half:]),
            frozen_seg[:half],
            u_prefix,
            phi,
            offset,
            f_op,
        )
        return rec(
            g_operation(llr[:half], llr[half:], u1_up),
            frozen_seg[half:],
            offset + half,
        )

    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    return rec(np.asarray(llr_ch, dtype=np.float64), frozen_bits, 0)
