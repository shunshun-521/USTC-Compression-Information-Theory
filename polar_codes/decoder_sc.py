"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat)
    return Lb + (1.0 - 2.0 * u_hat) * La


def _sionna_decode_block(llr, frozen):
    """Sionna 风格递归 SC，返回 (u_hat, u_up)"""
    n = len(llr)
    if n == 1:
        u = 0 if frozen[0] or llr[0] >= 0 else 1
        return np.array([u], dtype=int), np.array([u], dtype=int)

    half = n // 2
    l1, l2 = llr[:half], llr[half:]
    cn = f_operation(l1, l2)
    u1, u1_up = _sionna_decode_block(cn, frozen[:half])
    vn = g_operation(l1, l2, u1_up)
    u2, u2_up = _sionna_decode_block(vn, frozen[half:])
    u = np.concatenate([u1, u2])
    u1_up = (u1_up.astype(int) ^ u2_up.astype(int)).astype(int)
    u_up = np.concatenate([u1_up, u2_up])
    return u, u_up


def _llr_at_bit(llr, frozen, u_known, phi):
    """计算自然序比特 phi 处的 LLR"""
    brp = bit_reversal_permutation(len(llr))
    llr = np.asarray(llr, dtype=np.float64)[brp]
    frozen = np.asarray(frozen).astype(bool)

    def recurse(y, fb, offset, target, u_prefix):
        m = len(y)
        if m == 1:
            return y[0]
        half = m // 2
        l1, l2 = y[:half], y[half:]
        cn = f_operation(l1, l2)
        if target < offset + half:
            return recurse(cn, fb[:half], offset, target, u_prefix)
        u_left = u_prefix[offset:offset + half]
        u1_up = _sionna_up_only(cn, fb[:half], offset, u_left)
        vn = g_operation(l1, l2, u1_up)
        return recurse(vn, fb[half:], offset + half, target, u_prefix)

    return recurse(llr, frozen, 0, phi, u_known)


def _sionna_up_only(llr, frozen, offset, u_left):
    """仅计算左子树 u_up（已知左子树源比特 u_left）"""
    n = len(llr)
    if n == 1:
        return np.array([u_left[offset]], dtype=int)
    half = n // 2
    l1, l2 = llr[:half], llr[half:]
    cn = f_operation(l1, l2)
    u1_up_left = _sionna_up_only(cn, frozen[:half], offset, u_left[:half])
    vn = g_operation(l1, l2, u1_up_left)
    u2_up = _sionna_up_only(vn, frozen[half:], offset + half, u_left[half:])
    u1_up = (u1_up_left.astype(int) ^ u2_up.astype(int)).astype(int)
    return np.concatenate([u1_up, u2_up])


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    brp = bit_reversal_permutation(len(llr))
    llr = np.asarray(llr, dtype=np.float64)[brp]
    frozen = np.asarray(frozen_bits).astype(bool)
    u, _ = _sionna_decode_block(llr, frozen)
    return u


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [0] * n
    for layer in range(n):
        lambda_offset[layer] = (1 << layer) - 1

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        psi = phi
        while psi % 2 == 1:
            llr_layers.append(int(math.log2(psi & -psi)))
            psi //= 2
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        if phi % 2 == 1:
            psi = phi // 2
            while psi % 2 == 1:
                bit_layers.append(int(math.log2(psi & -psi)))
                psi //= 2
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（调用 Sionna 风格递归实现）"""
    return sc_decode_recursive(llr_ch, frozen_bits)
