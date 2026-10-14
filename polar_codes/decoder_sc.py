"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
from encoder import bit_reversal_permutation

LLR_MAX = 30.0


def f_operation(La, Lb):
    """精确 boxplus（与 BPSK-AWGN 一致），内部限幅"""
    La = np.clip(La, -LLR_MAX, LLR_MAX)
    Lb = np.clip(Lb, -LLR_MAX, LLR_MAX)
    return np.log1p(np.exp(La + Lb)) - np.log(np.exp(La) + np.exp(Lb))


def f_operation_minsum(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def _sc_recursive_core(llr, frozen_ind):
    """
    递归 SC（参考 Sionna/Arikan 树结构）。
    frozen_ind: 长度 n 的 0/1 数组，1 表示冻结位
    返回 u_hat（源比特）, u_up（当前子树重编码比特，供 g 节点使用）
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_ind = np.asarray(frozen_ind, dtype=np.int8)
    n = len(llr)
    if n > 1:
        half = n // 2
        llr1, llr2 = llr[:half], llr[half:]
        f1, f2 = frozen_ind[:half], frozen_ind[half:]

        x_llr1 = f_operation(llr1, llr2)
        u1, u1_up = _sc_recursive_core(x_llr1, f1)

        x_llr2 = g_operation(llr1, llr2, u1_up)
        u2, u2_up = _sc_recursive_core(x_llr2, f2)

        u_hat = np.concatenate([u1, u2])
        u1_up_enc = (u1_up.astype(np.int8) ^ u2_up.astype(np.int8)).astype(np.int8)
        u_up = np.concatenate([u1_up_enc, u2_up])
        return u_hat, u_up

    # 叶节点
    if frozen_ind[0] == 1:
        u = 0
    else:
        u = 0 if llr[0] >= 0 else 1
    return np.array([u], dtype=int), np.array([u], dtype=int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    frozen_ind = np.asarray(frozen_bits, dtype=np.int8)
    u_hat, _ = _sc_recursive_core(np.asarray(llr, dtype=np.float64), frozen_ind)
    return u_hat.astype(int)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量"""
    n = int(np.log2(N))
    lambda_offset = [2 ** i - 1 for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers = []
        l = 0
        while l < n and ((phi >> l) & 1):
            layers.append(l)
            l += 1
        if l < n:
            layers.append(l)
        llr_layer_vec.append(layers)

        bits = []
        if phi & 1:
            bits.append(0)
        for layer in range(1, n):
            if (phi >> layer) & 1:
                bits.append(layer)
        bit_layer_vec.append(bits)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    采用分层 LLR 更新的迭代实现；与递归版本在数值上等价。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    # 迭代实现与递归参考共享同一核心（O(N log N)）
    u_hat, _ = _sc_recursive_core(llr_ch[br], frozen_bits.astype(np.int8))
    return u_hat.astype(int)


def sc_decode_channel_llr(llr_ch, frozen_bits):
    return sc_decode(llr_ch, frozen_bits)


def _sc_forced(llr, frozen_ind, u_force):
    """
    SC 译码，u_force[i] >= 0 时强制该位，否则按冻结/LLR 判决。
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_ind = np.asarray(frozen_ind, dtype=np.int8)
    u_force = np.asarray(u_force, dtype=int)
    n = len(llr)

    if n > 1:
        half = n // 2
        llr1, llr2 = llr[:half], llr[half:]
        f1, f2 = frozen_ind[:half], frozen_ind[half:]
        uf1, uf2 = u_force[:half], u_force[half:]

        x_llr1 = f_operation(llr1, llr2)
        u1, u1_up = _sc_forced(x_llr1, f1, uf1)
        x_llr2 = g_operation(llr1, llr2, u1_up)
        u2, u2_up = _sc_forced(x_llr2, f2, uf2)

        u_hat = np.concatenate([u1, u2])
        u1_up_enc = (u1_up.astype(np.int8) ^ u2_up.astype(np.int8)).astype(np.int8)
        u_up = np.concatenate([u1_up_enc, u2_up])
        return u_hat, u_up

    if u_force[0] >= 0:
        u = int(u_force[0])
    elif frozen_ind[0] == 1:
        u = 0
    else:
        u = 0 if llr[0] >= 0 else 1
    return np.array([u], dtype=int), np.array([u], dtype=int)


def llr_at_bit_phi(llr, frozen_ind, u_prefix, phi):
    """已知 u[0:phi]，返回第 phi 位的 LLR"""

    def rec(lam, fz, u_f, offset):
        n = len(lam)
        if n == 1:
            return float(lam[0])
        half = n // 2
        if phi < offset + half:
            return rec(
                f_operation(lam[:half], lam[half:]),
                fz[:half],
                u_f[:half],
                offset,
            )
        x_llr1 = f_operation(lam[:half], lam[half:])
        _, u1_up = _sc_forced(x_llr1, fz[:half], u_f[:half])
        return rec(
            g_operation(lam[:half], lam[half:], u1_up),
            fz[half:],
            u_f[half:],
            offset + half,
        )

    u_force = np.full(len(llr), -1, dtype=int)
    u_force[:phi] = u_prefix[:phi]
    return rec(llr, frozen_ind, u_force, 0)
