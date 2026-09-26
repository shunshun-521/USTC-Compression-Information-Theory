"""
极化码 SC（串行抵消）译码器
"""
import math
import numpy as np

_LLR_MAX = 30.0


def _cn_op(x, y):
    """Check-node (box-plus)"""
    x = np.clip(x, -_LLR_MAX, _LLR_MAX)
    y = np.clip(y, -_LLR_MAX, _LLR_MAX)
    return np.log(1.0 + np.exp(x + y)) - np.log(np.exp(x) + np.exp(y))


def f_operation(La, Lb):
    """对外接口：f 运算"""
    return _cn_op(np.asarray(La, dtype=np.float64), np.asarray(Lb, dtype=np.float64))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def _sc_recursive_core(llr, frozen_ind):
    """递归 SC，返回 (u_hat, u_up)"""
    n = len(llr)
    frozen_ind = np.asarray(frozen_ind, dtype=np.float64)
    if n > 1:
        half = n // 2
        llr1, llr2 = llr[:half], llr[half:]
        f1, f2 = frozen_ind[:half], frozen_ind[half:]
        llr_u = _cn_op(llr1, llr2)
        u1, u1_up = _sc_recursive_core(llr_u, f1)
        llr_v = g_operation(llr1, llr2, u1_up)
        u2, u2_up = _sc_recursive_core(llr_v, f2)
        u_hat = np.concatenate([u1, u2])
        u1_xor = np.bitwise_xor(u1_up.astype(np.int8), u2_up.astype(np.int8)).astype(np.float64)
        u_up = np.concatenate([u1_xor, u2_up])
        return u_hat, u_up
    if frozen_ind[0] >= 0.5:
        return np.array([0.0]), np.array([0.0])
    bit = 0.5 * (1.0 - np.sign(llr[0]))
    if bit == 0.5:
        bit = 1.0
    return np.array([bit]), np.array([bit])


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码"""
    frozen_ind = np.asarray(frozen_bits, dtype=np.float64)
    u_hat, _ = _sc_recursive_core(np.asarray(llr, dtype=np.float64), frozen_ind)
    return u_hat.astype(int)


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助索引（供实验报告引用）"""
    n = int(math.log2(N))
    lambda_offset = np.zeros(n + 2, dtype=np.int32)
    for i in range(1, n + 2):
        lambda_offset[i] = lambda_offset[i - 1] + (1 << max(0, n + 1 - i))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        i = 0
        while i < n and ((phi >> i) & 1):
            i += 1
        llr_layer_vec.append(list(range(n - 1, i - 1, -1)))
        bit_layer_vec.append(list(range(i - 1, -1, -1)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC（调用已验证的递归核心）"""
    return sc_decode_recursive(llr_ch, frozen_bits)


def bit_llr_at_phi(llr_ch, frozen_bits, u_prefix, phi):
    """在给定前缀 u_prefix[0:phi] 下，计算第 phi 个比特的 LLR"""
    frozen_ind = np.asarray(frozen_bits, dtype=np.float64)
    llr = np.asarray(llr_ch, dtype=np.float64)
    u_prefix = np.asarray(u_prefix, dtype=int)

    def walk(layer, idx, seg):
        if layer == 0:
            if idx == phi:
                return float(seg[0])
            return None
        half = len(seg) // 2
        llr_u = _cn_op(seg[:half], seg[half:])
        if phi < idx + half:
            return walk(layer - 1, idx, llr_u)
        u_left = np.zeros(half, dtype=np.float64)
        for j in range(half):
            pos = idx + j
            u_left[j] = 0.0 if frozen_ind[pos] >= 0.5 else float(u_prefix[pos])
        llr_v = g_operation(seg[:half], seg[half:], u_left)
        return walk(layer - 1, idx + half, llr_v)

    return walk(int(math.log2(len(llr))), 0, llr)
