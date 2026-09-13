"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（委托递归实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """f 运算（log-domain box-plus）。"""
    La = np.atleast_1d(np.asarray(La, dtype=np.float64))
    Lb = np.atleast_1d(np.asarray(Lb, dtype=np.float64))
    large = (np.abs(La) > 30) | (np.abs(Lb) > 30)
    result = np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))
    small = ~large
    if np.any(small):
        ta = np.tanh(La[small] / 2.0)
        tb = np.tanh(Lb[small] / 2.0)
        prod = np.clip(ta * tb, -1.0 + 1e-15, 1.0 - 1e-15)
        result = result.copy()
        result[small] = 2.0 * np.arctanh(prod)
    return result


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat, dtype=int)
    return (1 - 2 * u_hat) * La + Lb


def _polar_decode_sc(llr_ch, frozen_ind):
    """
    递归 SC 译码（Sionna/Stimming LLR 域实现）。
    frozen_ind: 长度 n 的 bool 数组，True 表示冻结位。
    返回 u_hat, u_hat_up。
    """
    n = len(llr_ch)
    if n > 1:
        half = n // 2
        llr1 = llr_ch[:half]
        llr2 = llr_ch[half:]
        frozen1 = frozen_ind[:half]
        frozen2 = frozen_ind[half:]

        x_llr1 = f_operation(llr1, llr2)
        u_hat1, u_hat1_up = _polar_decode_sc(x_llr1, frozen1)

        x_llr2 = g_operation(llr1, llr2, u_hat1_up)
        u_hat2, u_hat2_up = _polar_decode_sc(x_llr2, frozen2)

        u_hat = np.concatenate([u_hat1, u_hat2])
        u_hat_up = np.concatenate([(u_hat1_up + u_hat2_up) % 2, u_hat2_up])
        return u_hat, u_hat_up

    is_frozen = frozen_ind[0]
    if is_frozen:
        u_hat = np.array([0], dtype=int)
    else:
        llr_val = llr_ch[0]
        u_hat = np.array([0 if llr_val >= 0 else 1], dtype=int)
    return u_hat, u_hat.copy()


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    u_hat, _ = _polar_decode_sc(np.asarray(llr, dtype=np.float64), frozen_bits)
    return u_hat.astype(int)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码。"""
    return sc_decode_recursive(llr_ch, frozen_bits)


def sc_decode_with_llrs(llr_ch, frozen_bits):
    """SC 译码并返回每位 LLR（供 SCL 使用）。"""
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    llr = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)
    bit_llrs = np.zeros(N, dtype=np.float64)

    def recurse(node_llr, frozen_node, offset):
        n = len(node_llr)
        if n == 1:
            idx = offset
            bit_llrs[idx] = node_llr[0]
            if frozen_node[0]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if node_llr[0] >= 0 else 1
            return u_hat[idx:idx + 1], u_hat[idx:idx + 1]

        half = n // 2
        left_llr = f_operation(node_llr[:half], node_llr[half:])
        u1, u1_up = recurse(left_llr, frozen_node[:half], offset)
        right_llr = g_operation(node_llr[:half], node_llr[half:], u1_up)
        u2, u2_up = recurse(right_llr, frozen_node[half:], offset + half)
        u_comb = np.concatenate([u1, u2])
        u_up = np.concatenate([(u1_up + u2_up) % 2, u2_up])
        return u_comb, u_up

    recurse(llr, frozen_bits, 0)
    return u_hat, bit_llrs


def get_bit_llr(llr_ch, frozen_bits, u_hat, phi):
    """获取第 phi 个比特的 LLR（供 SCL 使用，简化实现）。"""
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    partial = u_hat.copy()
    llr = np.asarray(llr_ch, dtype=np.float64)

    def llr_at(node_llr, frozen_node, offset, target):
        n = len(node_llr)
        if n == 1:
            return node_llr[0]
        half = n // 2
        if target < offset + half:
            return llr_at(f_operation(node_llr[:half], node_llr[half:]), frozen_node[:half], offset, target)
        u_up = partial[offset:offset + half]
        return llr_at(
            g_operation(node_llr[:half], node_llr[half:], u_up),
            frozen_node[half:], offset + half, target,
        )

    return llr_at(llr, frozen_bits, 0, phi)


def precompute_sc_indices(N):
    """保留接口兼容性。"""
    n = int(math.log2(N))
    lambda_offset = [0] * (n + 1)
    for layer in range(1, n + 1):
        lambda_offset[layer] = lambda_offset[layer - 1] + (1 << (n - layer))
    return lambda_offset, [[] for _ in range(N)], [[] for _ in range(N)]


def path_metric_update(pm, llr_val, u_decided):
    """路径度量更新。"""
    hard = 0 if llr_val >= 0 else 1
    if u_decided != hard:
        pm += abs(llr_val)
    return pm
